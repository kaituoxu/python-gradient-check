import numpy as np

# unidirectional Compact vFSMN
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
  - hidden  : TxD
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


# bidirectional Compact vFSMN
def bi_compact_vfsmn_memory_forward(hidden, bfilter, ffilter, bposition, fposition):
  """
  Only implement memory computation formula.
  NOTE: this implementation is `not` based on the vFSMN open source implementation,
        it's based on formulas in FSMN paper.
  Parameters:
  - hidden   : T    x D; input
  - bfilter  : N1+1 x D; weight matrix, N1 is look-back order, +1 means plusing current frame
  - ffilter  : N2   x D; weight matrix, N2 is lookahead order
  - bposition: T       ; auxiliary information; b means look backward
  - fposition: T       ; f means look forward
  Return:
  - memory   : T x D; output
  """
  T, D = hidden.shape
  N1 = bfilter.shape[0] - 1  # NOTE! must -1
  N2 = ffilter.shape[0]

  memory = np.zeros_like(hidden)
  for r in range(T):
    for c in range(D):
      # main implementation, check this part with CUDA kernel implementation
      # lookback
      step = min(bposition[r], N1)
      result = hidden[r, c] + bfilter[0, c] * hidden[r, c]
      for i in range(step):
        result += bfilter[i + 1, c] * hidden[r - i - 1, c]
      # lookahead
      step = min(fposition[r], N2)
      for i in range(step):
        result += ffilter[i, c] * hidden[r + i + 1, c]
      # done
      memory[r, c] = result
  return memory


def bi_compact_vfsmn_memory_backward(dmemory, hidden, bfilter, ffilter, bposition, fposition):
  """
  Parameters:
  - dmemory  : T    x D
  - hidden   : T    x D
  - bfilter  : N1+1 x D
  - ffilter  : N2   x D
  - bposition: T
  - fposition: T
  Returns:
  - dhidden  : T    x D
  - dbfilter : N1+1 x D
  - dffilter : N2   x D
  """
  T, D = dmemory.shape
  N1 = bfilter.shape[0] - 1  # NOTE! must -1
  N2 = ffilter.shape[0]

  dhidden = np.zeros_like(dmemory)
  # _compute_vfsmn_hidden_diff
  for r in range(T):
    for c in range(D):
      # main implementation, check this part with CUDA kernel implementation
      # lookback
      step = min(bposition[r], N1)
      dhidden[r, c] += dmemory[r, c] + bfilter[0, c] * dmemory[r, c]  ### NOTE HERE!!!
      for i in range(step):
        dhidden[r - i - 1, c] += bfilter[i + 1, c] * dmemory[r, c]
      # lookahead
      step = min(fposition[r], N2)
      for i in range(step):
        dhidden[r + i + 1, c] += ffilter[i, c] * dmemory[r, c]

  dbfilter = np.zeros_like(bfilter)
  dffilter = np.zeros_like(ffilter)
  # _update_vfsmn_filter
  for r in range(T):
    for c in range(D):
      # main implementation, check this part with CUDA kernel implementation
      # lookback
      step = min(bposition[r], N1)
      dbfilter[0, c] += hidden[r, c] * dmemory[r, c]
      for i in range(step):
        dbfilter[i + 1, c] += hidden[r - i - 1, c] * dmemory[r, c]
      # lookahead
      step = min(fposition[r], N2)
      for i in range(step):
        dffilter[i, c] += hidden[r + i + 1, c] * dmemory[r, c]

  return dhidden, dbfilter, dffilter
