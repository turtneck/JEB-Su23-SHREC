# ParallelMazeSolver
ECSE 420 project for Fall 2019 Semester, McGill University - Parallel Maze Solver with Cuda v10.1.

The purpose of this project is to navigate through a maze with the help of parallel computing. The program starts at the top left corner of a given maze and attempts to find a path to the bottom right corner. The simplest method for finding this path is using recursion to backtrack from the end to the start, similar to depth first search. However, recursion consumes a lot of time and space when performed.

One simple optimization is to use memoization to ensure that we check each grid in the maze only once. This would help us reduce our runtime complexity to O(rc). Even with this optimization, we would have multiple recursive calls and we would have to wait for the previous branches of the recursive tree to return from its deepest node before we can try a different path. Therefore, a parallel approach is needed to explore multiple possible paths at the same time.

The full report can be found [here](https://github.com/pierrerm/ParallelMazeSolver/blob/master/FinalReport.pdf "Final Report").

## Results:

#### Speedup

![alt text](https://github.com/pierrerm/ParallelMazeSolver/blob/master/Maze/speedup.png "speedup")

#### 101x101 maze

![alt text](https://github.com/pierrerm/ParallelMazeSolver/blob/master/Maze/maze101_p.png "101 maze")

#### 401x401 maze

![alt text](https://github.com/pierrerm/ParallelMazeSolver/blob/master/Maze/maze401_p.png "401 maze")

#### 1001x1001 maze

![alt text](https://github.com/pierrerm/ParallelMazeSolver/blob/master/Maze/maze1001_p.png "1001 maze")

## Authors:  
   Stefano Commodari 260742659 <br>
   Anudruth Manjunath 260710646 <br>
   Pierre Robert-Michon 260712449 <br>
   Romain Couperier 260724748 <br>

