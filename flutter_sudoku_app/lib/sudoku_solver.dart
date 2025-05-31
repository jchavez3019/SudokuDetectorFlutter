/// Description: Given a pixel coordinate, return the index into its
/// corresponding grid. Since there are 9 grid, the returned index should be in
/// the range [0, 8]. This check is not enforced.
///
/// **Parameters**:
/// - `i`: The i'th row the pixel is located at, should be in range [0, 8]
/// - `j`: The j'th row the pixel is located at, should be in range [0, 8]
int gbmIdx(int i, int j) {
  return (i ~/ 3) * 3 + (j ~/ 3);
}

/// Description: Given a list of integers and the index into this list, check to
/// see if the bit in position `num` is set to 1.
///
/// **Parameters**:
/// - num: The bit position to check in the integer.
/// - idx: The index into the bitset.
/// - bitset: List of integers.
int isBitSet(int num, int idx, List<int> bitset) {
  return (bitset[idx] >> num) & 0x0001;
}

/// Description: Recursive function that aids the main function for solving a
/// sudoku puzzle. Performs backtracking once infeasibility is detected.
///
/// **Parameters**:
/// - k: The current cell in the board, in range [0, 80].
/// - board: The current sudoku board to modify in place.
/// - rbm:  The row bit mask. 1 indicates an element has been visited at the
///         corresponding row.
/// - cbm:  The column bit mask. 1 indicates an element has been visited at the
///         corresponding column.
/// - gbm:  The grid bit mask. 1 indicates an element has been visited at the
///         corresponding grid.
/// - is_fixed: Element is marked true if its position is fixed on the board.
///
/// **Returns**:
/// - True if values can be placed in cells [k, 80] while maintaining feasibility.
bool _backtracking(int k, List<List<String>> board, List<int> rbm,
    List<int> cbm, List<int> gbm, List<bool> is_fixed) {
  if (k >= 81) {
    // base case, we are out of range on the board
    return true;
  }
  if (is_fixed[k]) {
    // this is a fixed element, move on to the next
    return _backtracking(k+1, board, rbm, cbm, gbm, is_fixed);
  }
  int i = k ~/ 9;
  int j = k % 9;
  int g = gbmIdx(i, j);
  for (int c = 0; c < 9; c++) {
    // iterate through sudoku digits
    if ((isBitSet(c, i, rbm) | isBitSet(c, j, cbm) | isBitSet(c, g, gbm)) == 0) {
      // this element has not been visited in any row, column, or grid; try it
      rbm[i] |= 0x0001 << c;
      cbm[j] |= 0x0001 << c;
      gbm[g] |= 0x0001 << c;
      if (_backtracking(k+1, board, rbm, cbm, gbm, is_fixed)) {
        // setting this element does not make subsequent entries infeasible
        board[i][j] = "${c+1}";
        return true;
      } else {
        // this element creates an infeasible sudoku board, reset our decision
        rbm[i] &= ~(0x0001 << c);
        cbm[j] &= ~(0x0001 << c);
        gbm[g] &= ~(0x0001 << c);
      }
    }
  }
  // no viable options, this board is not feasible
  return false;
}

/// Description: Solves a sudoku board in place.
///
/// **Parameters**:
/// - board: The initial sudoku board which will get modified in place.
///
/// **Returns**:
/// - True if the board contains a solution, False otherwise.
bool solveSudoku(List<List<String>> board) {
  List<int> rbm = List.generate(9, (_) => 0);
  List<int> cbm = List.generate(9, (_) => 0);
  List<int> gbm = List.generate(9, (_) => 0);
  List<bool> is_fixed = List.generate(81, (_) => false);

  // fill in the bit masks and is_fixed vector for fixed elements on the board
  int i, j, g, c;
  String char;
  for (int k = 0; k < 81; k++) {
    i = k ~/9;
    j = k % 9;
    char = board[i][j];
    if (char != '.') {
      g = gbmIdx(i, j);
      c = int.parse(char) - 1;
      is_fixed[k] = true;
      rbm[i] |= 0x0001 << c;
      cbm[j] |= 0x0001 << c;
      gbm[g] |= 0x0001 << c;
    }
  }

  return _backtracking(0, board, rbm, cbm, gbm, is_fixed);
}

//// Description: As a standalone file, we may pass a sudoku puzzle to solve with
//// its corresponding solution board to check the correctness of the methods in
//// this file.
// void main(List<String> arguments) {
//   List<List<String>> board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]];
//   List<List<String>> expected = [["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]];
//   print("Original board: \n${board}");
//   print("Solution board: \n${expected}");
//
//   bool has_solution = solveSudoku(board);
//
//   print("Output: \n${board}");
//
//   if (has_solution) {
//     bool outputs_match = true;
//     for (int i = 0; i < 9; i++) {
//       for (int j = 0; j < 9; j++) {
//         if (board[i][j] != expected[i][j]) {
//           outputs_match = false;
//           break;
//         }
//       }
//       if (!outputs_match) {
//         break;
//       }
//     }
//     String match_string = outputs_match ? "match" : "does not match";
//     print("The returned board and solution board ${match_string}.");
//   }
//   else {
//     print("The board did not have a solution!");
//   }
// }