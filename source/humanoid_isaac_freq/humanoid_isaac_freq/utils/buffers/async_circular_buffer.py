"""Per-environment circular buffer used by asynchronous sensors."""

from collections.abc import Sequence

import torch
from isaaclab.utils.buffers import CircularBuffer


class AsyncCircularBuffer(CircularBuffer):
    """A :class:`CircularBuffer` with independent pointers per environment."""

    def __init__(self, max_len: int, batch_size: int, device: str):
        super().__init__(max_len, batch_size, device)

    @property
    def buffer(self) -> torch.Tensor:
        if torch.any(self._num_pushes == 0):
            raise RuntimeError(
                "Attempting to access a buffer that is not fully initialized."
            )
        return self.get_by_batch_ids()

    def get_by_batch_ids(
        self, batch_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return selected histories in chronological order (oldest to newest)."""
        if self._buffer is None:
            raise RuntimeError(
                "Attempting to retrieve data from an empty circular buffer."
            )
        selected_ids = (
            self._ALL_INDICES
            if batch_ids is None
            else torch.as_tensor(batch_ids, device=self._device)
        )
        if torch.any(self._num_pushes[selected_ids] == 0):
            raise RuntimeError(
                "Attempting to retrieve data on an empty circular buffer."
            )

        shifts = self.max_length - self._pointer[selected_ids] - 1
        selected_buf = self._buffer[:, selected_ids, ...].clone()
        selected_batch_size = selected_ids.numel()
        history_length = self.max_length
        history_indices = torch.arange(history_length, device=self._device)
        gather_indices = (
            (history_indices[:, None] - shifts[None, :]) % history_length
        ).long()
        extra_shape = selected_buf.shape[2:]
        gather_indices = gather_indices.view(
            history_length, selected_batch_size, *([1] * len(extra_shape))
        ).expand(history_length, selected_batch_size, *extra_shape)
        chronological = torch.gather(selected_buf, dim=0, index=gather_indices)
        return chronological.transpose(0, 1)

    def append(
        self,
        data: torch.Tensor,
        batch_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Append samples for all or selected environments."""
        selected_ids = (
            self._ALL_INDICES
            if batch_ids is None
            else torch.as_tensor(batch_ids, device=self._device)
        )
        if data.shape[0] != len(selected_ids):
            raise ValueError(
                f"Data batch {data.shape[0]} does not match batch_ids length {len(selected_ids)}."
            )

        data = data.to(self._device)
        if self._buffer is None:
            self._pointer = -torch.ones(
                self._batch_size, dtype=torch.int, device=self._device
            )
            self._buffer = torch.empty(
                (self.max_length, self._batch_size, *data.shape[1:]),
                device=self._device,
                dtype=data.dtype,
            )

        self._pointer[selected_ids] = (
            self._pointer[selected_ids] + 1
        ) % self.max_length
        self._buffer[self._pointer[selected_ids], selected_ids] = data
        is_first_push = self._num_pushes[selected_ids] == 0
        if torch.any(is_first_push):
            first_ids = selected_ids[is_first_push]
            self._buffer[:, first_ids] = data[is_first_push]
        self._num_pushes[selected_ids] += 1

    def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset selected histories and their pointers."""
        selected_ids = (
            self._ALL_INDICES
            if batch_ids is None
            else torch.as_tensor(batch_ids, device=self._device)
        )
        self._num_pushes[selected_ids] = 0
        if isinstance(self._pointer, torch.Tensor):
            self._pointer[selected_ids] = -1
        if self._buffer is not None:
            self._buffer[:, selected_ids] = 0.0
