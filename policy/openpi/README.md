# MobiAgent policy backend

This is the OpenPI-derived backend extracted from the S1 deployment and A100
training workspaces. It contains the shared VLM, routed flow-matching experts,
normalization, data loaders, training, and WebSocket serving code.

See [the training guide](../../docs/training.md) for installation and entrypoints.
The upstream Apache-2.0 license and Gemma notice are retained in this directory.
Historical experiment configurations and generated assets are excluded.
