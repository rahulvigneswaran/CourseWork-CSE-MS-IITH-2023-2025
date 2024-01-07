# Test cases and checklists

## With initial feasible point
- `testcase_1.csv` : With initial feasible point, Non-degenerate, Bounded. https://www.desmos.com/calculator/3pscsowphb

- `testcase_2.csv` : With initial feasible point, Non-degenerate, Unbounded. https://www.desmos.com/calculator/mgfg9jhofq

- `testcase_3.csv` : With inital feasible point, Degenerate, Bounded. https://www.desmos.com/calculator/06uocynnxe
- `testcase_4.csv` : With initial feasible point, Degenerate, Unbounded. https://www.desmos.com/calculator/04h97moxit

## Without initial feasible point
- `testcase_5.csv` : Without inital feasible point, Non-degenerate, Bounded. All positive b. SAME AS testcase_1.csv but without the 1st row. https://www.desmos.com/calculator/3pscsowphb
- `testcase_6.csv` : Without inital feasible point, Non-degenerate, Bounded. Atleast one -ve b. https://www.desmos.com/calculator/zwfsdzbrk0

- `testcase_7.csv` : Without inital feasible point, Non-degenerate, Unbounded. All positive b. SAME AS testcase_2.csv but without the 1st row. https://www.desmos.com/calculator/mgfg9jhofq

- `testcase_8.csv` : Without inital feasible point, Non-degenerate, Unbounded. Atleast one -ve b. https://www.desmos.com/calculator/kboknzaxln

- `testcase_9.csv` : Without inital feasible point, Degenerate, Bounded. All positive b. SAME AS testcase_3.csv but without the 1st row. https://www.desmos.com/calculator/06uocynnxe
- `testcase_10.csv` : Without inital feasible point, Degenerate, Bounded. Atleast one -ve b. https://www.desmos.com/calculator/hkczrfpjny

- `testcase_11.csv` : Without inital feasible point, Degenerate, Unbounded. All positive b. SAME AS testcase_4.csv but without the 1st row. https://www.desmos.com/calculator/04h97moxit
- `testcase_12.csv` : Without inital feasible point, Degenerate, Unbounded. Atleast one -ve b. https://www.desmos.com/calculator/ix2fiu9gva

|                 | **A1** | **A2** | **A3** | **A4** |
|-----------------|--------|--------|--------|--------|
| **Test Case 1** | PASS   | PASS   | PASS   | -      |
| **Test Case 2** | -      | PASS   | PASS   | -      |
| **Test Case 3** | -      | -      | PASS   | -      |
| **Test Case 4** | -      | -      | PASS   | -      |
| **Test Case 5** | -      | -      | -      | PASS   |
| **Test Case 6** | -      | -      | -      | PASS   |
| **Test Case 7** | -      | -      | -      | PASS   |
| **Test Case 8** | -      | -      | -      | PASS   |
| **Test Case 9** | -      | -      | -      | PASS   |
| **Test Case 10**| -      | -      | -      | PASS   |
| **Test Case 11**| -      | -      | -      | PASS   |
| **Test Case 12**| -      | -      | -      | PASS   |

`-`: Not applicable
`PASS`: The test case passed for the corresponding assignements.