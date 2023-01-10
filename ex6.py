import numpy as np

def simplex(c):
  # Initialize basic variables
  m = len(c)
  basic_vars = list(range(m))
  nonbasic_vars = []
  tableau = np.eye(m)
  cbar = c
  
  while True:
    # Find entering variable
    j = np.argmin(cbar[nonbasic_vars])
    j = nonbasic_vars[j]
    if cbar[j] >= 0:
      return basic_vars, np.zeros(m)
    
    # Find leaving variable
    i = np.argmin(tableau[basic_vars, j])
    i = basic_vars[i]
    if tableau[i, j] <= 0:
      return None, None
    
    # Pivot
    tableau[i, :] /= tableau[i, j]
    for k in range(m):
      if k != i and abs(tableau[k, j]) > 1e-10:
        tableau[k, :] -= tableau[k, j] * tableau[i, :]
    
    # Update basic and nonbasic variables
    basic_vars[i] = j
    nonbasic_vars[j] = i


if __name__ == "__main__":
    f = np.array([])