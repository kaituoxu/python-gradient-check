import numpy as np

# _vfsmn_memory
def compact_vfsmn_memory_forward(hidden, filter, position):
  """
  Only implement memory computation formula.
  NOTE: this implementation is based on the vFSMN open source implementation,
        it's a little different with formulas in FSMN paper.
  TODO: make this same with FSMN paper.
  Parameters:
  - hidden   : T x D; input
  - filter   : N x D; weight matrix, N is look-back order
  - position : T; auxiliary information
  Return:
  - memory   : T x D; output
  """
  T, D = hidden.shape
  N = filter.shape[0]
  memory = np.zeros((T, D))
  for r in range(T):
    for c in range(D):
      # main implementation, check this part with CUDA kernel implementation
      step = min(position[r], N)
      result = hidden[r, c]
      for i in range(step):
        result += filter[i, c] * hidden[r - i - 1, c]
      memory[r, c] = result
  return memory


def compact_vfsmn_memory_backward(dmemory, hidden, filter, position):
  """
  Parameters:
  - dmemory : TxD
  - filter  : NxD
  - position: T
  """
  T, D = dmemory.shape
  N = filter.shape[0]
  dhidden = np.zeros_like(dmemory)
  # _compute_vfsmn_hidden_diff
  for r in range(T):
    for c in range(D):
      # main implementation, check this part with CUDA kernel implementation
      step = min(position[r], N)
      dhidden[r, c] += dmemory[r, c]  ### NOTE HERE!!!
      for i in range(step):
        dhidden[r-i-1, c] += filter[i, c] * dmemory[r, c]

  dfilter = np.zeros_like(filter)
  # _update_vfsmn_filter
  for r in range(T):
    for c in range(D):
      # main implementation, check this part with CUDA kernel implementation
      step = min(position[r], N)
      for i in range(step):
        dfilter[i, c] += hidden[r - i - 1, c] * dmemory[r, c]

  return dhidden, dfilter
