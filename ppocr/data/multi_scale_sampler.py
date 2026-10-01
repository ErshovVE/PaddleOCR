from paddle.io import Sampler
import paddle.distributed as dist

import numpy as np
import random
import math


class MultiScaleSampler(Sampler):
    def __init__(
        self,
        data_source,
        scales,
        first_bs=128,
        fix_bs=True,
        divided_factor=[8, 16],
        is_training=True,
        ratio_wh=0.8,
        max_w=480.0,
        seed=None,
        sorted_batch_ratio=1.0,
        pad_to_longest=False,
    ):
        """
        multi scale samper
        Args:
            data_source(dataset)
            scales(list): several scales for image resolution
            first_bs(int): batch size for the first scale in scales
            divided_factor(list[w, h]): ImageNet models down-sample images by a factor, ensure that width and height dimensions are multiples are multiple of devided_factor.
            is_training(boolean): mode. When False (e.g. in Eval) the data is not
                split between cards, every card sees the whole dataset.
            max_w(float): upper bound of the batch width when ds_width is True.
            sorted_batch_ratio(float): only with ds_width. Share of batches made of
                images with similar aspect ratio (sorted by w/h); the rest are
                made of randomly shuffled images. 1.0 keeps the legacy behaviour
                (all batches sorted). Batches are re-drawn every epoch when < 1.0.
            pad_to_longest(bool): only with ds_width. Batch width follows the
                widest image of the batch (capped by max_w) instead of the mean
                ratio, so long lines are not squeezed. Shuffled batches then get
                their own width too instead of the fixed scale width. The dataset
                rounds the ratio, so the widest image may still be squeezed by
                less than one image height.
        """
        assert (
            0.0 <= sorted_batch_ratio <= 1.0
        ), "sorted_batch_ratio must be in [0, 1], got {}".format(sorted_batch_ratio)
        # min. and max. spatial dimensions
        self.data_source = data_source
        self.data_idx_order_list = np.array(data_source.data_idx_order_list)
        self.ds_width = data_source.ds_width
        self.seed = data_source.seed
        if self.ds_width:
            self._sync_wh()
        self.n_data_samples = len(self.data_source)
        assert sorted_batch_ratio == 1.0 or self.ds_width, (
            "sorted_batch_ratio < 1 requires dataset ds_width: true "
            "(batches are sorted by width/height ratio)"
        )
        self.ratio_wh = ratio_wh
        self.max_w = max_w
        self.sorted_batch_ratio = sorted_batch_ratio
        self.pad_to_longest = pad_to_longest
        self._mix_epoch = 0

        if isinstance(scales[0], list):
            width_dims = [i[0] for i in scales]
            height_dims = [i[1] for i in scales]
        elif isinstance(scales[0], int):
            width_dims = scales
            height_dims = scales
        base_im_w = width_dims[0]
        base_im_h = height_dims[0]
        base_batch_size = first_bs

        # Get the GPU and node related information
        if is_training:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        else:
            num_replicas, rank = 1, 0
        # adjust the total samples to avoid batch dropping
        num_samples_per_replica = int(self.n_data_samples * 1.0 / num_replicas)

        img_indices = [idx for idx in range(self.n_data_samples)]

        self.shuffle = False
        if is_training:
            # compute the spatial dimensions and corresponding batch size
            # ImageNet models down-sample images by a factor of 32.
            # Ensure that width and height dimensions are multiples are multiple of 32.
            width_dims = [
                int((w // divided_factor[0]) * divided_factor[0]) for w in width_dims
            ]
            height_dims = [
                int((h // divided_factor[1]) * divided_factor[1]) for h in height_dims
            ]

            img_batch_pairs = list()
            base_elements = base_im_w * base_im_h * base_batch_size
            for h, w in zip(height_dims, width_dims):
                if fix_bs:
                    batch_size = base_batch_size
                else:
                    batch_size = int(max(1, (base_elements / (h * w))))
                img_batch_pairs.append((w, h, batch_size))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
        else:
            self.img_batch_pairs = [(base_im_w, base_im_h, base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas

        self.batch_list = []
        self.current = 0
        last_index = num_samples_per_replica * num_replicas
        indices_rank_i = self.img_indices[self.rank : last_index : self.num_replicas]
        while self.current < self.n_samples_per_replica:
            for curr_w, curr_h, curr_bsz in self.img_batch_pairs:
                end_index = min(self.current + curr_bsz, self.n_samples_per_replica)
                batch_ids = indices_rank_i[self.current : end_index]
                n_batch_samples = len(batch_ids)
                if n_batch_samples != curr_bsz:
                    batch_ids += indices_rank_i[: (curr_bsz - n_batch_samples)]
                self.current += curr_bsz

                if len(batch_ids) > 0:
                    batch = [curr_w, curr_h, len(batch_ids)]
                    self.batch_list.append(batch)
        random.shuffle(self.batch_list)
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch = self.iter()
        self.batchs_in_one_epoch_id = [i for i in range(len(self.batchs_in_one_epoch))]

    def _sync_wh(self):
        """Take w/h of the dataset's current epoch (it changes with ratio_list)."""
        self.wh_ratio = self.data_source.wh_ratio
        self.wh_ratio_sort = self.data_source.wh_ratio_sort
        # position of every image in the ratio-sorted order
        self.sort_pos = np.argsort(self.wh_ratio_sort)
        self._wh_version = getattr(self.data_source, "wh_version", 0)

    def __iter__(self):
        wh_changed = self.ds_width and self._wh_version != getattr(
            self.data_source, "wh_version", 0
        )
        if wh_changed:
            # new per-epoch sample of the dataset (ratio_list < 1): re-plan batches
            self._sync_wh()
        if self.sorted_batch_ratio < 1.0 and self.shuffle:
            # new mix of sorted / shuffled batches every training epoch
            self._mix_epoch += 1
            self.batchs_in_one_epoch = self.iter()
        elif wh_changed:
            self.batchs_in_one_epoch = self.iter()
        if self.seed is None:
            random.seed(self.epoch)
            self.epoch += 1
        else:
            random.seed(self.seed)
        if self.shuffle:
            # evaluation (is_training=False) keeps a fixed batch order, so it
            # sees the same batches every time (Windows skips the last one)
            random.shuffle(self.batchs_in_one_epoch_id)
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batchs_in_one_epoch[batch_tuple_id]

    def iter(self):
        if self.ds_width and self.sorted_batch_ratio < 1.0 and self.shuffle:
            return self._iter_mixed()
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            else:
                random.seed(self.epoch)
            if not self.ds_width:
                random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_pairs)
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]
        else:
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]

        start_index = 0
        batchs_in_one_epoch = []
        for batch_tuple in self.batch_list:
            curr_w, curr_h, curr_bsz = batch_tuple
            end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != curr_bsz and self.shuffle:
                # training fills the last batch up; evaluation keeps it short so
                # that no sample is counted twice
                batch_ids += indices_rank_i[: (curr_bsz - n_batch_samples)]
            start_index += curr_bsz

            if len(batch_ids) > 0:
                if self.ds_width:
                    wh_ratio_current = self.wh_ratio[self.wh_ratio_sort[batch_ids]]
                    ratio_current = self._batch_ratio(wh_ratio_current, curr_h)
                else:
                    ratio_current = None
                batch = [(curr_w, curr_h, b_id, ratio_current) for b_id in batch_ids]
                # yield batch
                batchs_in_one_epoch.append(batch)
        return batchs_in_one_epoch

    def _batch_ratio(self, wh_ratios, curr_h):
        ratio = wh_ratios.max() if self.pad_to_longest else wh_ratios.mean()
        return ratio if ratio * curr_h < self.max_w else self.max_w / curr_h

    def _iter_mixed(self):
        """Batches of two kinds: sorted by w/h ratio and randomly shuffled.

        Every image is used once per epoch: a shuffled permutation is split, its
        head feeds the shuffled batches and the tail, sorted by ratio, feeds the
        sorted ones. Tuples are (w, h, idx, ratio, is_sorted): sorted batches
        pass the position in the ratio-sorted order, shuffled ones the index in
        data_idx_order_list.
        """
        seed = (0 if self.seed is None else self.seed) + self._mix_epoch
        rng = random.Random(seed)
        # batch_list is shuffled differently on every rank; a canonical order
        # makes all ranks draw the same split (the batch order is shuffled
        # later in __iter__ anyway)
        batch_list = sorted(self.batch_list)
        use_sorted = [rng.random() < self.sorted_batch_ratio for _ in batch_list]
        n_shuffled = sum(
            bsz for (_, _, bsz), flag in zip(batch_list, use_sorted) if not flag
        )
        n_shuffled = min(n_shuffled * self.num_replicas, len(self.img_indices))
        perm = list(self.img_indices)
        rng.shuffle(perm)
        streams = {
            False: perm[:n_shuffled][self.rank :: self.num_replicas],
            True: sorted(int(self.sort_pos[i]) for i in perm[n_shuffled:])[
                self.rank :: self.num_replicas
            ],
        }
        starts = {False: 0, True: 0}

        batchs_in_one_epoch = []
        for (curr_w, curr_h, curr_bsz), is_sorted in zip(batch_list, use_sorted):
            if not streams[is_sorted]:
                # tiny dataset: one kind of batch got no images at all
                is_sorted = not is_sorted
            stream, start = streams[is_sorted], starts[is_sorted]
            batch_ids = stream[start : start + curr_bsz]
            if len(batch_ids) < curr_bsz:
                # stream exhausted: fill up from its own start, as legacy does
                batch_ids += stream[: curr_bsz - len(batch_ids)]
            starts[is_sorted] = start + curr_bsz
            if is_sorted:
                file_ids = self.wh_ratio_sort[batch_ids]
                ratio = self._batch_ratio(self.wh_ratio[file_ids], curr_h)
            elif self.pad_to_longest:
                file_ids = self.data_idx_order_list[batch_ids]
                ratio = self._batch_ratio(self.wh_ratio[file_ids], curr_h)
            else:
                ratio = None
            batchs_in_one_epoch.append(
                [(curr_w, curr_h, b_id, ratio, is_sorted) for b_id in batch_ids]
            )
        return batchs_in_one_epoch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length
