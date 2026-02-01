# Computer Science Core Books Breakdown

---

## 1. Structure and Interpretation of Computer Programs (SICP)

### Book Overview

| Detail | Info |
|--------|------|
| Authors | Harold Abelson, Gerald Jay Sussman |
| Pages | ~600 |
| Language | Scheme (Lisp dialect) |
| Difficulty | Challenging |
| Free Online | Yes - mitpress.mit.edu/sites/default/files/sicp/full-text/book/book.html |

### What It Teaches
Programming as a way of thinking, not just coding. Focuses on abstraction, recursion, and building complex systems from simple parts.

### Chapter Breakdown

<details>
<summary><strong>Chapter 1: Building Abstractions with Procedures</strong></summary>

**Topics:**
- Elements of programming
- Procedures and processes
- Recursion vs iteration
- Orders of growth
- Higher-order procedures

**Key Concepts:**
```scheme
; Recursive factorial
(define (factorial n)
  (if (= n 1)
      1
      (* n (factorial (- n 1)))))

; Iterative factorial
(define (factorial n)
  (define (iter product counter)
    (if (> counter n)
        product
        (iter (* counter product) (+ counter 1))))
  (iter 1 1))

; Higher-order: procedures as arguments
(define (sum term a next b)
  (if (> a b)
      0
      (+ (term a)
         (sum term (next a) next b))))
```

**Exercises to Do:** 1.3, 1.7, 1.8, 1.11, 1.12, 1.16, 1.17, 1.19, 1.29, 1.31, 1.32, 1.33, 1.37, 1.38, 1.41, 1.42, 1.43, 1.44, 1.46

**Time:** 2-3 weeks

</details>

<details>
<summary><strong>Chapter 2: Building Abstractions with Data</strong></summary>

**Topics:**
- Data abstraction
- Hierarchical data (pairs, lists, trees)
- Symbolic data
- Multiple representations
- Generic operations

**Key Concepts:**
```scheme
; Data abstraction - rational numbers
(define (make-rat n d) (cons n d))
(define (numer x) (car x))
(define (denom x) (cdr x))

; List operations
(define (map proc items)
  (if (null? items)
      nil
      (cons (proc (car items))
            (map proc (cdr items)))))

; Tree recursion
(define (count-leaves tree)
  (cond ((null? tree) 0)
        ((not (pair? tree)) 1)
        (else (+ (count-leaves (car tree))
                 (count-leaves (cdr tree))))))
```

**Exercises to Do:** 2.1, 2.2, 2.3, 2.4, 2.5, 2.17, 2.18, 2.19, 2.20, 2.21, 2.23, 2.27, 2.28, 2.29, 2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.40, 2.41, 2.42

**Time:** 3-4 weeks

</details>

<details>
<summary><strong>Chapter 3: Modularity, Objects, and State</strong></summary>

**Topics:**
- Assignment and local state
- Environment model
- Mutable data
- Concurrency
- Streams

**Key Concepts:**
```scheme
; Object with local state
(define (make-account balance)
  (define (withdraw amount)
    (if (>= balance amount)
        (begin (set! balance (- balance amount))
               balance)
        "Insufficient funds"))
  (define (deposit amount)
    (set! balance (+ balance amount))
    balance)
  (define (dispatch m)
    (cond ((eq? m 'withdraw) withdraw)
          ((eq? m 'deposit) deposit)
          (else (error "Unknown request"))))
  dispatch)

; Streams (lazy evaluation)
(define (stream-map proc s)
  (if (stream-null? s)
      the-empty-stream
      (cons-stream (proc (stream-car s))
                   (stream-map proc (stream-cdr s)))))
```

**Exercises to Do:** 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.8, 3.12, 3.13, 3.14, 3.16, 3.17, 3.18, 3.21, 3.22, 3.23, 3.50, 3.51, 3.52, 3.53, 3.54, 3.55, 3.56

**Time:** 3-4 weeks

</details>

<details>
<summary><strong>Chapter 4: Metalinguistic Abstraction</strong></summary>

**Topics:**
- The metacircular evaluator
- Lazy evaluation
- Nondeterministic computing
- Logic programming

**Key Concepts:**
```scheme
; Core of the evaluator
(define (eval exp env)
  (cond ((self-evaluating? exp) exp)
        ((variable? exp) (lookup-variable-value exp env))
        ((quoted? exp) (text-of-quotation exp))
        ((assignment? exp) (eval-assignment exp env))
        ((definition? exp) (eval-definition exp env))
        ((if? exp) (eval-if exp env))
        ((lambda? exp) (make-procedure (lambda-parameters exp)
                                       (lambda-body exp)
                                       env))
        ((application? exp)
         (apply (eval (operator exp) env)
                (list-of-values (operands exp) env)))
        (else (error "Unknown expression type"))))

(define (apply procedure arguments)
  (cond ((primitive-procedure? procedure)
         (apply-primitive-procedure procedure arguments))
        ((compound-procedure? procedure)
         (eval-sequence
           (procedure-body procedure)
           (extend-environment
             (procedure-parameters procedure)
             arguments
             (procedure-environment procedure))))
        (else (error "Unknown procedure type"))))
```

**Exercises to Do:** 4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.11, 4.13, 4.14, 4.15, 4.16

**Time:** 4-5 weeks

</details>

<details>
<summary><strong>Chapter 5: Computing with Register Machines</strong></summary>

**Topics:**
- Register machine design
- A register-machine simulator
- Storage allocation and garbage collection
- Compilation

**Key Concepts:**
- How abstract programs map to machine operations
- Explicit control structure
- Compilation vs interpretation

**Time:** 2-3 weeks (can skim or skip if time-constrained)

</details>

### How to Study SICP

1. **Set up Scheme:** Use Racket with `#lang sicp` or MIT Scheme
2. **Type every example:** Don't just read
3. **Do the exercises:** At minimum, do the ones marked above
4. **Draw environment diagrams:** Essential for Chapter 3
5. **Build the interpreter:** Chapter 4 is the payoff

### Alternative
*How to Design Programs* (HtDP) - more accessible, uses Racket, better for complete beginners

---

## 2. Computer Systems: A Programmer's Perspective (CS:APP)

### Book Overview

| Detail | Info |
|--------|------|
| Authors | Randal Bryant, David O'Hallaron |
| Pages | ~1000 |
| Language | C, some assembly |
| Difficulty | Moderate to Challenging |
| Edition | 3rd (2015) recommended |

### What It Teaches
How computer systems execute programs, from bits to processes. Bridges gap between high-level programming and hardware.

### Chapter Breakdown

<details>
<summary><strong>Chapter 1: A Tour of Computer Systems</strong></summary>

**Topics:**
- Compilation system overview
- Hardware organization
- Caches matter
- Storage hierarchy
- OS abstractions

**Key Takeaways:**
- Understand the full path from source code to execution
- Memory hierarchy fundamentally shapes performance
- OS provides abstractions over hardware

**Time:** 1-2 days (overview chapter)

</details>

<details>
<summary><strong>Chapter 2: Representing and Manipulating Information</strong></summary>

**Topics:**
- Information storage (bytes, words)
- Integer representations (unsigned, two's complement)
- Integer arithmetic
- Floating point

**Key Concepts:**
```c
// Two's complement range for w bits
// Min: -2^(w-1)
// Max: 2^(w-1) - 1

// For 8 bits:
// Range: -128 to 127

// Overflow example
int x = INT_MAX;  // 2147483647
x + 1;            // -2147483648 (wraps around)

// Floating point: (-1)^s × M × 2^E
// Sign bit, mantissa, exponent
// Not all decimals representable exactly
0.1 + 0.2 != 0.3  // In floating point!
```

**Exercises to Do:** 2.1, 2.3, 2.4, 2.10, 2.11, 2.12, 2.13, 2.23, 2.24, 2.25, 2.27, 2.42, 2.49, 2.52

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 3: Machine-Level Representation of Programs</strong></summary>

**Topics:**
- Assembly basics (x86-64)
- Data formats and access
- Arithmetic and logical operations
- Control flow
- Procedures
- Arrays and structures
- Buffer overflow

**Key Concepts:**
```c
// C code
long mult2(long a, long b) {
    long s = a * b;
    return s;
}
```
```assembly
# x86-64 assembly
mult2:
    movq    %rdi, %rax    # a in %rdi
    imulq   %rsi, %rax    # multiply by b (in %rsi)
    ret                    # return value in %rax
```

**Register conventions:**
| Register | Purpose |
|----------|---------|
| %rdi | 1st argument |
| %rsi | 2nd argument |
| %rdx | 3rd argument |
| %rcx | 4th argument |
| %rax | Return value |
| %rsp | Stack pointer |
| %rbp | Base pointer |

**Exercises to Do:** 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.9, 3.23, 3.24, 3.29, 3.58, 3.59, 3.60, 3.61, 3.62, 3.63

**Labs:** Bomb Lab (highly recommended), Attack Lab

**Time:** 3-4 weeks

</details>

<details>
<summary><strong>Chapter 4: Processor Architecture</strong></summary>

**Topics:**
- Y86-64 instruction set
- Logic design basics
- Sequential implementation
- Pipelining
- Pipeline hazards

**Key Concepts:**
```
Pipeline stages:
1. Fetch: Read instruction from memory
2. Decode: Read registers
3. Execute: ALU operation
4. Memory: Read/write memory
5. Write back: Write to register
6. PC update: Update program counter

Hazards:
- Data hazard: Instruction needs result from previous
- Control hazard: Branch not yet resolved
- Solutions: Stalling, forwarding, branch prediction
```

**Time:** 2-3 weeks (can skim if focusing on software)

</details>

<details>
<summary><strong>Chapter 5: Optimizing Program Performance</strong></summary>

**Topics:**
- Compiler capabilities and limitations
- Expressing program performance
- Loop unrolling
- Parallelism
- Memory performance

**Key Concepts:**
```c
// Original
for (i = 0; i < n; i++)
    sum += a[i];

// Unrolled (2x)
for (i = 0; i < n-1; i += 2) {
    sum += a[i];
    sum += a[i+1];
}
if (i < n) sum += a[i];

// Parallel accumulators
for (i = 0; i < n-1; i += 2) {
    sum0 += a[i];
    sum1 += a[i+1];
}
sum = sum0 + sum1;
```

**Exercises to Do:** 5.5, 5.6, 5.13, 5.14, 5.15, 5.16, 5.17

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 6: The Memory Hierarchy</strong></summary>

**Topics:**
- Storage technologies
- Locality
- Cache memories
- Cache-friendly code

**Key Concepts:**
```
Cache organization:
- S = 2^s sets
- E = lines per set
- B = 2^b bytes per block

Address breakdown: [Tag | Set Index | Block Offset]

Types:
- Direct mapped: E = 1
- Set associative: 1 < E < C/B
- Fully associative: E = C/B

Cache-friendly patterns:
- Sequential access (stride-1)
- Repeated access to same data
- Working set fits in cache
```

```c
// Cache-friendly (row-major access)
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        sum += a[i][j];

// Cache-unfriendly (column-major access)
for (j = 0; j < N; j++)
    for (i = 0; i < N; i++)
        sum += a[i][j];
```

**Exercises to Do:** 6.2, 6.3, 6.4, 6.5, 6.8, 6.9, 6.10, 6.11, 6.12, 6.13, 6.17, 6.18, 6.19, 6.20

**Labs:** Cache Lab

**Time:** 2-3 weeks

</details>

<details>
<summary><strong>Chapter 7: Linking</strong></summary>

**Topics:**
- Compiler drivers
- Static linking
- Object files
- Symbols and symbol tables
- Relocation
- Dynamic linking
- Position-independent code

**Key Concepts:**
```
Compilation pipeline:
source.c → [Preprocessor] → source.i
         → [Compiler] → source.s
         → [Assembler] → source.o
         → [Linker] → executable

Symbol types:
- Global: Defined here, visible externally
- External: Referenced here, defined elsewhere
- Local: Defined and used only here (static)
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 8: Exceptional Control Flow</strong></summary>

**Topics:**
- Exceptions
- Processes
- System calls
- Process control (fork, exec, wait)
- Signals
- Nonlocal jumps

**Key Concepts:**
```c
// Fork creates new process
pid_t pid = fork();
if (pid == 0) {
    // Child process
    printf("Child PID: %d\n", getpid());
    exit(0);
} else {
    // Parent process
    printf("Parent PID: %d, Child: %d\n", getpid(), pid);
    wait(NULL);  // Wait for child
}

// Signal handling
void handler(int sig) {
    printf("Caught signal %d\n", sig);
}

signal(SIGINT, handler);  // Ctrl-C handler
```

**Exercises to Do:** 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.9, 8.10, 8.11, 8.12, 8.13, 8.15, 8.16, 8.18, 8.20, 8.22, 8.24, 8.25

**Labs:** Shell Lab

**Time:** 2-3 weeks

</details>

<details>
<summary><strong>Chapter 9: Virtual Memory</strong></summary>

**Topics:**
- Physical and virtual addressing
- Address translation
- Page tables
- TLB
- Memory mapping
- Dynamic memory allocation (malloc)
- Garbage collection

**Key Concepts:**
```
Virtual address → [TLB] → Physical address
                  miss ↓
              [Page Table]
                  miss ↓
              [Page Fault]
              [Disk Access]

Address translation:
Virtual:  [VPN | VPO]  (Virtual Page Number, Offset)
Physical: [PPN | PPO]  (Physical Page Number, Offset)
VPO = PPO (page offset unchanged)
```

```c
// malloc/free
void *p = malloc(100);  // Allocate 100 bytes
free(p);                // Deallocate

// Common bugs:
// - Memory leak (forget to free)
// - Double free
// - Use after free
// - Buffer overflow
```

**Exercises to Do:** 9.1, 9.2, 9.3, 9.4, 9.5, 9.11, 9.12, 9.13, 9.14, 9.15, 9.17, 9.18, 9.19, 9.20

**Labs:** Malloc Lab

**Time:** 2-3 weeks

</details>

<details>
<summary><strong>Chapters 10-12: System I/O, Networking, Concurrency</strong></summary>

**Chapter 10 - System I/O:**
- Unix I/O
- Files and file descriptors
- Reading and writing files
- Robust I/O

**Chapter 11 - Network Programming:**
- Client-server model
- Sockets interface
- Web servers

**Chapter 12 - Concurrent Programming:**
- Threads
- Synchronization (semaphores, mutexes)
- Thread safety
- Parallelism

**Time:** 2-3 weeks total

**Labs:** Proxy Lab (combines networking and concurrency)

</details>

### Labs Summary

| Lab | Chapter | Skills |
|-----|---------|--------|
| Data Lab | 2 | Bit manipulation |
| Bomb Lab | 3 | Assembly reading, GDB |
| Attack Lab | 3 | Buffer overflow exploits |
| Cache Lab | 6 | Cache simulation, optimization |
| Shell Lab | 8 | Process control, signals |
| Malloc Lab | 9 | Memory allocation |
| Proxy Lab | 10-12 | Networking, concurrency |

### How to Study CS:APP

1. **Do the labs** - They're the most valuable part
2. **Use GDB extensively** - Essential for Bomb Lab
3. **Write C code** - Practice memory management
4. **Measure performance** - Use `perf`, `valgrind`
5. **Read assembly** - Get comfortable with x86-64

---

## 3. The Algorithm Design Manual

### Book Overview

| Detail | Info |
|--------|------|
| Author | Steven Skiena |
| Pages | ~750 |
| Language | C, pseudocode |
| Difficulty | Moderate |
| Edition | 3rd (2020) recommended |

### What It Teaches
Practical algorithm design with focus on real-world problem solving. Half textbook, half catalog of algorithmic problems.

### Part I: Practical Algorithm Design

<details>
<summary><strong>Chapter 1: Introduction to Algorithm Design</strong></summary>

**Topics:**
- Robot tour optimization
- Selecting the right jobs
- Reasoning about correctness
- Modeling the problem

**Key Takeaways:**
- Correct algorithms vs heuristics
- Problem modeling is crucial
- Counterexamples disprove algorithms

**Time:** 2-3 days

</details>

<details>
<summary><strong>Chapter 2: Algorithm Analysis</strong></summary>

**Topics:**
- RAM model of computation
- Big-O notation
- Growth rates
- Dominance relations
- Reasoning about efficiency

**Key Concepts:**
```
Growth rates (slowest to fastest):
1 < log n < √n < n < n log n < n² < n³ < 2^n < n!

Big-O: Upper bound (worst case)
Big-Ω: Lower bound (best case)  
Big-Θ: Tight bound (exact)

f(n) = O(g(n)) means f(n) ≤ c·g(n) for large n
```

**Exercises:** 2-1 through 2-10, 2-35 through 2-45

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 3: Data Structures</strong></summary>

**Topics:**
- Contiguous vs linked structures
- Stacks and queues
- Dictionaries
- Binary search trees
- Priority queues
- Hashing

**Key Concepts:**
```
Array vs Linked List:
                Array    Linked List
Access          O(1)     O(n)
Insert/Delete   O(n)     O(1)*
Memory          Fixed    Dynamic

*If you have pointer to location

Dictionary implementations:
- Unsorted array: O(n) search, O(1) insert
- Sorted array: O(log n) search, O(n) insert
- Hash table: O(1) average all ops
- BST: O(log n) average all ops
```

**Exercises:** 3-1 through 3-10, 3-21 through 3-30

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 4: Sorting</strong></summary>

**Topics:**
- Applications of sorting
- Heapsort
- Mergesort
- Quicksort
- Distribution sort
- Lower bounds

**Key Concepts:**
```python
# Mergesort - O(n log n), stable
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

# Quicksort - O(n log n) average, O(n²) worst
def quicksort(arr, lo, hi):
    if lo < hi:
        p = partition(arr, lo, hi)
        quicksort(arr, lo, p - 1)
        quicksort(arr, p + 1, hi)

# Comparison sort lower bound: Ω(n log n)
```

| Algorithm | Best | Average | Worst | Space | Stable |
|-----------|------|---------|-------|-------|--------|
| Mergesort | n log n | n log n | n log n | O(n) | Yes |
| Quicksort | n log n | n log n | n² | O(log n) | No |
| Heapsort | n log n | n log n | n log n | O(1) | No |

**Exercises:** 4-1 through 4-15, 4-36 through 4-45

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 5: Divide and Conquer</strong></summary>

**Topics:**
- Binary search
- Recurrence relations
- Master theorem
- Fast multiplication
- Matrix multiplication

**Key Concepts:**
```
Master Theorem for T(n) = aT(n/b) + f(n):

1. If f(n) = O(n^(log_b(a) - ε)) → T(n) = Θ(n^log_b(a))
2. If f(n) = Θ(n^log_b(a)) → T(n) = Θ(n^log_b(a) · log n)
3. If f(n) = Ω(n^(log_b(a) + ε)) → T(n) = Θ(f(n))

Examples:
T(n) = 2T(n/2) + n     → O(n log n)  [Mergesort]
T(n) = T(n/2) + 1      → O(log n)    [Binary search]
T(n) = 2T(n/2) + 1     → O(n)        [Tree traversal]
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 6: Hashing and Randomized Algorithms</strong></summary>

**Topics:**
- Hash functions
- Collision resolution
- Bloom filters
- Randomized algorithms

**Key Concepts:**
```
Collision resolution:
1. Chaining: Linked list at each bucket
2. Open addressing: Probe for next empty slot
   - Linear probing: h(k) + i
   - Quadratic probing: h(k) + i²
   - Double hashing: h(k) + i·h'(k)

Load factor α = n/m (items/buckets)
Expected chain length = α
Keep α < 0.75, resize when exceeded
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 7: Graph Traversal</strong></summary>

**Topics:**
- Graph representations
- BFS and DFS
- Applications of traversal
- Connected components
- Topological sorting

**Key Concepts:**
```python
# BFS - shortest path in unweighted graph
def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        for neighbor in graph[v]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# DFS - traversal, cycle detection
def dfs(graph, v, visited):
    visited.add(v)
    for neighbor in graph[v]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Topological sort (DFS-based)
def topo_sort(graph):
    visited = set()
    order = []
    
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
        order.append(v)
    
    for v in graph:
        if v not in visited:
            dfs(v)
    
    return order[::-1]
```

**Exercises:** 7-1 through 7-15, 7-21 through 7-30

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 8: Weighted Graph Algorithms</strong></summary>

**Topics:**
- Minimum spanning trees (Prim, Kruskal)
- Shortest paths (Dijkstra, Bellman-Ford, Floyd-Warshall)
- Network flows

**Key Concepts:**
```python
# Dijkstra's algorithm - O((V+E) log V) with heap
def dijkstra(graph, start):
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    
    return dist

# Bellman-Ford - handles negative edges, O(VE)
def bellman_ford(edges, n, start):
    dist = [float('inf')] * n
    dist[start] = 0
    
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None  # Negative cycle
    
    return dist
```

| Algorithm | Graph Type | Complexity |
|-----------|------------|------------|
| Dijkstra | Non-negative weights | O((V+E) log V) |
| Bellman-Ford | Any (detects negative cycles) | O(VE) |
| Floyd-Warshall | All pairs | O(V³) |
| Prim/Kruskal | MST | O(E log V) |

**Exercises:** 8-1 through 8-15, 8-21 through 8-30

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 9: Combinatorial Search</strong></summary>

**Topics:**
- Backtracking
- Pruning search
- Constraint satisfaction

**Key Concepts:**
```python
# Backtracking template
def backtrack(state):
    if is_solution(state):
        process_solution(state)
        return
    
    for candidate in get_candidates(state):
        if is_valid(candidate, state):
            add_candidate(state, candidate)
            backtrack(state)
            remove_candidate(state, candidate)  # Backtrack

# N-Queens example
def solve_nqueens(n):
    def backtrack(row, cols, diag1, diag2, board):
        if row == n:
            solutions.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            if col in cols or row-col in diag1 or row+col in diag2:
                continue
            
            board[row][col] = 'Q'
            backtrack(row+1, cols|{col}, diag1|{row-col}, 
                     diag2|{row+col}, board)
            board[row][col] = '.'
    
    solutions = []
    board = [['.']*n for _ in range(n)]
    backtrack(0, set(), set(), set(), board)
    return solutions
```

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 10: Dynamic Programming</strong></summary>

**Topics:**
- Caching vs computation
- Approximate string matching
- Longest increasing subsequence
- Knapsack problem
- Matrix chain multiplication

**Key Concepts:**
```python
# DP Framework:
# 1. Define state: dp[i] = optimal answer for subproblem i
# 2. Recurrence: dp[i] = f(dp[j]) for j < i
# 3. Base case: dp[0] = ...
# 4. Order: Fill table in correct order
# 5. Answer: Return dp[n] or similar

# Edit distance (Levenshtein)
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # delete
                                  dp[i][j-1],    # insert
                                  dp[i-1][j-1])  # replace
    
    return dp[m][n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity+1) for _ in range(n+1)]
    
    for i in range(1, n+1):
        for w in range(capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w],
                              dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

**Exercises:** 10-1 through 10-15

**Time:** 2-3 weeks (most important chapter)

</details>

<details>
<summary><strong>Chapter 11: NP-Completeness</strong></summary>

**Topics:**
- Problems vs algorithms
- Reductions
- P, NP, NP-complete, NP-hard
- Classic NP-complete problems

**Key Concepts:**
```
P: Solvable in polynomial time
NP: Verifiable in polynomial time
NP-complete: Hardest problems in NP
NP-hard: At least as hard as NP-complete

If ANY NP-complete problem in P → P = NP

To prove problem X is NP-complete:
1. Show X ∈ NP (solution verifiable in poly time)
2. Reduce known NP-complete problem to X

Classic NP-complete problems:
- SAT (satisfiability)
- 3-SAT
- Vertex Cover
- Independent Set
- Clique
- Hamiltonian Path/Cycle
- Traveling Salesman
- Graph Coloring
- Subset Sum
- Knapsack
```

**Time:** 1 week

</details>

### Part II: The Hitchhiker's Guide to Algorithms

Chapters 12-21 form a catalog of algorithmic problems. Use as reference:

| Chapter | Topic |
|---------|-------|
| 12 | Data Structures |
| 13 | Numerical Problems |
| 14 | Combinatorial Problems |
| 15 | Graph Problems: Polynomial |
| 16 | Graph Problems: Hard |
| 17 | Computational Geometry |
| 18 | Set and String Problems |
| 19 | Algorithmic Resources |

### How to Study Algorithm Design Manual

1. **Read Part I thoroughly** - Do exercises
2. **Implement key algorithms** from scratch
3. **Practice on LeetCode** - Apply patterns
4. **Use Part II as reference** - When you encounter a problem type
5. **Watch Skiena's lectures** - Great supplement

---

## 4. Mathematics for Computer Science (MIT 6.042J)

### Book Overview

| Detail | Info |
|--------|------|
| Authors | Eric Lehman, Tom Leighton, Albert Meyer |
| Pages | ~1000 |
| Difficulty | Moderate |
| Free Online | Yes - MIT OpenCourseWare |

### What It Teaches
Mathematical foundations for CS: proofs, discrete structures, counting, probability, and graph theory.

### Part I: Proofs

<details>
<summary><strong>Chapter 1: What is a Proof?</strong></summary>

**Topics:**
- Propositions
- Predicates
- Axioms and inference rules
- Proof methods

**Key Concepts:**
```
Proof methods:
1. Direct proof: Assume P, show Q
2. Proof by contrapositive: Prove ¬Q → ¬P
3. Proof by contradiction: Assume ¬P, derive contradiction
4. Proof by cases: Split into exhaustive cases
```

**Example - Direct Proof:**
```
Theorem: If n is odd, then n² is odd.

Proof:
- Assume n is odd
- Then n = 2k + 1 for some integer k
- n² = (2k + 1)² = 4k² + 4k + 1 = 2(2k² + 2k) + 1
- This is of form 2m + 1, so n² is odd ∎
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 2: The Well Ordering Principle</strong></summary>

**Topics:**
- Well ordering principle
- Template for WOP proofs

**Key Concept:**
```
Well Ordering Principle:
Every nonempty set of nonnegative integers has a smallest element.

Proof template:
1. Define set C of counterexamples
2. Assume C is nonempty
3. By WOP, C has minimum element m
4. Reach contradiction (usually by finding smaller counterexample)
5. Therefore C is empty ∎
```

**Time:** 3-4 days

</details>

<details>
<summary><strong>Chapter 3: Logical Formulas</strong></summary>

**Topics:**
- Propositional logic
- Truth tables
- Logical equivalences
- SAT and validity

**Key Concepts:**
```
Connectives:
- NOT: ¬P
- AND: P ∧ Q
- OR: P ∨ Q
- IMPLIES: P → Q  (equivalent to ¬P ∨ Q)
- IFF: P ↔ Q

Important equivalences:
- De Morgan: ¬(P ∧ Q) ≡ ¬P ∨ ¬Q
- De Morgan: ¬(P ∨ Q) ≡ ¬P ∧ ¬Q
- Contrapositive: (P → Q) ≡ (¬Q → ¬P)
- Implication: (P → Q) ≡ (¬P ∨ Q)
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 4: Mathematical Data Types</strong></summary>

**Topics:**
- Sets
- Sequences
- Functions
- Binary relations

**Key Concepts:**
```
Set operations:
- Union: A ∪ B
- Intersection: A ∩ B
- Difference: A - B
- Complement: Ā
- Power set: P(A) = {all subsets of A}
- Cartesian product: A × B

Functions:
- Injective (one-to-one): f(a) = f(b) → a = b
- Surjective (onto): ∀b ∃a: f(a) = b
- Bijective: Both injective and surjective

Relations:
- Reflexive: aRa for all a
- Symmetric: aRb → bRa
- Transitive: aRb ∧ bRc → aRc
- Equivalence relation: All three above
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 5: Induction</strong></summary>

**Topics:**
- Ordinary induction
- Strong induction
- Structural induction

**Key Concepts:**
```
Ordinary Induction:
1. Base case: Prove P(0)
2. Inductive step: Prove P(n) → P(n+1)
3. Conclude: P(n) for all n ≥ 0

Strong Induction:
1. Base case: Prove P(0)
2. Inductive step: Prove [P(0) ∧ P(1) ∧ ... ∧ P(n)] → P(n+1)
3. Conclude: P(n) for all n ≥ 0

Use strong induction when P(n+1) depends on multiple previous values
```

**Example - Strong Induction:**
```
Theorem: Every integer n ≥ 2 is a product of primes.

Proof by strong induction:
Base: n = 2 is prime (product of one prime)

Inductive step: Assume true for all k where 2 ≤ k ≤ n
Show true for n+1:
- If n+1 is prime, done
- If n+1 is composite, n+1 = ab where 2 ≤ a,b ≤ n
- By IH, a and b are products of primes
- So n+1 = ab is product of primes ∎
```

**Exercises:** All problems in this chapter are important

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 6: State Machines</strong></summary>

**Topics:**
- State machines
- Invariants
- Derived variables

**Key Concept:**
```
Proving invariants:
1. Show property P holds in start state
2. Show if P holds before transition, P holds after
3. Conclude P holds in all reachable states
```

**Time:** 1 week

</details>

### Part II: Structures

<details>
<summary><strong>Chapters 7-10: Number Theory & Graphs</strong></summary>

**Chapter 7 - Infinite Sets:**
- Countable vs uncountable
- Cantor's diagonalization

**Chapter 8 - Number Theory:**
- Divisibility
- GCD and Euclidean algorithm
- Modular arithmetic
- RSA cryptography

**Chapter 9-10 - Graphs:**
- Graph definitions
- Connectivity
- Trees
- Coloring
- Planar graphs

**Key Concepts:**
```python
# Euclidean algorithm for GCD
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Extended Euclidean algorithm
def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

# Modular exponentiation
def mod_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod
    return result
```

**Time:** 3-4 weeks total

</details>

### Part III: Counting

<details>
<summary><strong>Chapters 11-14: Counting & Probability</strong></summary>

**Chapter 11 - Counting:**
- Sum and product rules
- Bijection rule
- Division rule

**Chapter 12 - Counting with Symmetries:**
- Permutations and combinations
- Binomial theorem
- Pigeonhole principle

**Chapter 13 - Recursion:**
- Recurrences
- Solving recurrences

**Chapter 14 - Infinite Sets:**
- Generating functions

**Key Formulas:**
```
Permutations (order matters):
P(n, k) = n! / (n-k)!

Combinations (order doesn't matter):
C(n, k) = n! / (k!(n-k)!)

Binomial theorem:
(x + y)^n = Σ C(n,k) x^k y^(n-k)

Stars and bars (k items in n bins):
C(n + k - 1, k)

Pigeonhole: If n+1 items in n bins, some bin has ≥ 2 items
```

**Time:** 3 weeks

</details>

### Part IV: Probability

<details>
<summary><strong>Chapters 15-19: Probability</strong></summary>

**Topics:**
- Sample spaces and events
- Conditional probability
- Independence
- Random variables
- Expected value
- Variance
- Deviation bounds

**Key Concepts:**
```
Basic probability:
P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
P(A | B) = P(A ∩ B) / P(B)

Independence:
P(A ∩ B) = P(A) · P(B)

Bayes' theorem:
P(A | B) = P(B | A) · P(A) / P(B)

Expected value:
E[X] = Σ x · P(X = x)
E[X + Y] = E[X] + E[Y]  (always true)
E[X · Y] = E[X] · E[Y]  (if independent)

Variance:
Var(X) = E[X²] - E[X]²
Var(X + Y) = Var(X) + Var(Y)  (if independent)

Common distributions:
- Bernoulli: E[X] = p, Var(X) = p(1-p)
- Binomial(n,p): E[X] = np, Var(X) = np(1-p)
- Geometric: E[X] = 1/p, Var(X) = (1-p)/p²
```

**Time:** 3-4 weeks

</details>

### How to Study MIT 6.042

1. **Watch the lectures** - Leighton is an excellent teacher
2. **Do the problem sets** - Available on OCW
3. **Work through proofs** - Don't skip steps
4. **Practice counting problems** - They're tricky
5. **Build intuition for probability** - Simulate if needed

---

## 5. Operating Systems: Three Easy Pieces (OSTEP)

### Book Overview

| Detail | Info |
|--------|------|
| Authors | Remzi Arpaci-Dusseau, Andrea Arpaci-Dusseau |
| Pages | ~700 |
| Difficulty | Moderate |
| Free Online | Yes - ostep.org |

### What It Teaches
How operating systems work: virtualization of CPU and memory, concurrency, and persistence.

### Part I: Virtualization

<details>
<summary><strong>Chapters 1-6: Processes</strong></summary>

**Topics:**
- Processes and process API
- Direct execution
- Scheduling (FIFO, SJF, STCF, RR, MLFQ)

**Key Concepts:**
```
Process states:
Running → Ready ← Blocked
         ↓   ↑
       Blocked

Process API (Unix):
- fork(): Create child process
- exec(): Replace process image
- wait(): Wait for child to finish

Context switch:
1. Save current process state (registers, PC)
2. Load new process state
3. Resume execution
```

**Scheduling algorithms:**

| Algorithm | Description | Pros | Cons |
|-----------|-------------|------|------|
| FIFO | First come, first served | Simple | Convoy effect |
| SJF | Shortest job first | Optimal average wait | Needs job length |
| STCF | Shortest time to completion | Preemptive SJF | Starvation possible |
| RR | Round robin (time slices) | Fair, good response | Poor turnaround |
| MLFQ | Multi-level feedback queue | Adaptive | Complex |

**Code:**
```c
// Fork example
int main() {
    pid_t pid = fork();
    
    if (pid < 0) {
        // Error
        fprintf(stderr, "Fork failed\n");
    } else if (pid == 0) {
        // Child
        printf("Child: PID = %d\n", getpid());
        execlp("/bin/ls", "ls", NULL);
    } else {
        // Parent
        wait(NULL);
        printf("Parent: Child finished\n");
    }
    return 0;
}
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapters 7-11: Scheduling</strong></summary>

**MLFQ Rules:**
```
1. If Priority(A) > Priority(B), A runs
2. If Priority(A) = Priority(B), A & B run in RR
3. New jobs enter at top priority
4. Once job uses time allotment at priority level, move down
5. After time period S, move all jobs to top queue (boost)
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapters 12-24: Memory Virtualization</strong></summary>

**Topics:**
- Address spaces
- Address translation
- Segmentation
- Free-space management
- Paging
- TLBs
- Multi-level page tables
- Swapping

**Key Concepts:**
```
Virtual memory goals:
1. Transparency: Process thinks it has all memory
2. Efficiency: Time and space
3. Protection: Isolation between processes

Address translation:
Virtual Address → [MMU] → Physical Address

Page table entry:
[Valid | Protection | Present | Dirty | Reference | PFN]

TLB (Translation Lookaside Buffer):
- Cache for address translations
- TLB hit: Fast (1 cycle)
- TLB miss: Slow (page table walk)

Multi-level page table:
- Saves space for sparse address spaces
- Only allocate page table pages as needed
```

**Page replacement algorithms:**

| Algorithm | Description |
|-----------|-------------|
| FIFO | Replace oldest page |
| Random | Replace random page |
| LRU | Replace least recently used |
| Clock | Approximate LRU (circular buffer) |

```c
// Memory mapping example
void *ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
// Use ptr...
munmap(ptr, 4096);
```

**Time:** 3-4 weeks

</details>

### Part II: Concurrency

<details>
<summary><strong>Chapters 25-34: Concurrency</strong></summary>

**Topics:**
- Threads
- Locks
- Condition variables
- Semaphores
- Common concurrency bugs
- Event-based concurrency

**Key Concepts:**
```c
// Thread creation
pthread_t thread;
pthread_create(&thread, NULL, thread_func, arg);
pthread_join(thread, NULL);

// Mutex
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_lock(&lock);
// Critical section
pthread_mutex_unlock(&lock);

// Condition variable
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_cond_wait(&cond, &lock);   // Wait (releases lock)
pthread_cond_signal(&cond);         // Wake one waiter
pthread_cond_broadcast(&cond);      // Wake all waiters
```

**Producer-Consumer:**
```c
int buffer[MAX];
int fill = 0, use = 0, count = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t empty = PTHREAD_COND_INITIALIZER;
pthread_cond_t full = PTHREAD_COND_INITIALIZER;

void *producer(void *arg) {
    for (int i = 0; i < loops; i++) {
        pthread_mutex_lock(&lock);
        while (count == MAX)
            pthread_cond_wait(&empty, &lock);
        buffer[fill] = i;
        fill = (fill + 1) % MAX;
        count++;
        pthread_cond_signal(&full);
        pthread_mutex_unlock(&lock);
    }
}

void *consumer(void *arg) {
    for (int i = 0; i < loops; i++) {
        pthread_mutex_lock(&lock);
        while (count == 0)
            pthread_cond_wait(&full, &lock);
        int tmp = buffer[use];
        use = (use + 1) % MAX;
        count--;
        pthread_cond_signal(&empty);
        pthread_mutex_unlock(&lock);
    }
}
```

**Semaphores:**
```c
sem_t sem;
sem_init(&sem, 0, initial_value);
sem_wait(&sem);   // Decrement, block if zero
sem_post(&sem);   // Increment, wake waiter
```

**Common bugs:**
- Atomicity violation: Check-then-act not atomic
- Order violation: A should happen before B, but doesn't
- Deadlock: Circular wait on locks

**Time:** 3-4 weeks

</details>

### Part III: Persistence

<details>
<summary><strong>Chapters 35-44: File Systems</strong></summary>

**Topics:**
- I/O devices
- Hard disk drives
- RAID
- File systems
- File system implementation
- FFS, journaling, LFS
- Flash-based SSDs

**Key Concepts:**
```
File system layers:
Application
    ↓
System call interface (open, read, write, close)
    ↓
Virtual File System (VFS)
    ↓
Specific file system (ext4, XFS, etc.)
    ↓
Block layer
    ↓
Device driver
    ↓
Hardware

Inode (index node):
- Metadata: size, permissions, timestamps
- Pointers to data blocks
- Direct, indirect, double indirect, triple indirect

Directory:
- List of (name, inode number) pairs
- Special entries: . (current), .. (parent)
```

**Journaling:**
```
Write-ahead logging:
1. Write transaction begin to journal
2. Write metadata/data to journal
3. Write transaction end to journal
4. Write data to actual location
5. Mark transaction complete

Crash recovery:
- Replay committed transactions
- Discard incomplete transactions
```

**File API:**
```c
int fd = open("file.txt", O_RDWR | O_CREAT, 0644);
read(fd, buffer, size);
write(fd, buffer, size);
lseek(fd, offset, SEEK_SET);
fsync(fd);  // Flush to disk
close(fd);
```

**Time:** 2-3 weeks

</details>

### How to Study OSTEP

1. **Read actively** - The writing is accessible
2. **Do the homework** - Available in ostep-homework repo
3. **Run the simulations** - They clarify concepts
4. **Study xv6** - Simple Unix implementation
5. **Write systems code** - Build a shell, implement malloc

---

## 6. Computer Networking: A Top-Down Approach

### Book Overview

| Detail | Info |
|--------|------|
| Authors | James Kurose, Keith Ross |
| Pages | ~800 |
| Difficulty | Moderate |
| Edition | 8th (2021) recommended |

### What It Teaches
How the Internet works, from applications down to physical links.

### Chapter Breakdown

<details>
<summary><strong>Chapter 1: Computer Networks and the Internet</strong></summary>

**Topics:**
- What is the Internet?
- Network edge and core
- Packet switching vs circuit switching
- Protocol layers
- Network attacks

**Key Concepts:**
```
Internet protocol stack:
5. Application (HTTP, DNS, SMTP)
4. Transport (TCP, UDP)
3. Network (IP)
2. Link (Ethernet, WiFi)
1. Physical (bits on wire)

Packet switching:
- Data split into packets
- Each packet routed independently
- Statistical multiplexing (share bandwidth)

Circuit switching:
- Dedicated path for duration of communication
- Guaranteed bandwidth
- Used in traditional phone networks
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 2: Application Layer</strong></summary>

**Topics:**
- HTTP
- Email (SMTP, IMAP)
- DNS
- P2P applications
- Video streaming
- Socket programming

**Key Concepts:**
```
HTTP request:
GET /index.html HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0
Connection: keep-alive

HTTP response:
HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 1234

<html>...

HTTP methods:
- GET: Retrieve resource
- POST: Submit data
- PUT: Update resource
- DELETE: Remove resource

DNS hierarchy:
Root servers → TLD servers (.com, .org) → Authoritative servers
                                              ↓
                                         IP address

DNS record types:
- A: Name → IPv4 address
- AAAA: Name → IPv6 address
- CNAME: Alias → canonical name
- MX: Domain → mail server
- NS: Domain → name server
```

**Socket programming:**
```python
# TCP Server
import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('', 12000))
server.listen(1)

while True:
    conn, addr = server.accept()
    data = conn.recv(1024)
    conn.send(data.upper())
    conn.close()

# TCP Client
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 12000))
client.send(b'hello')
response = client.recv(1024)
client.close()
```

**Time:** 2-3 weeks

</details>

<details>
<summary><strong>Chapter 3: Transport Layer</strong></summary>

**Topics:**
- Transport layer services
- UDP
- Reliable data transfer principles
- TCP
- Congestion control

**Key Concepts:**
```
UDP header (8 bytes):
[Source Port | Dest Port | Length | Checksum]
- Connectionless
- No reliability
- No congestion control
- Fast, simple

TCP header (20+ bytes):
[Source Port | Dest Port]
[Sequence Number]
[Acknowledgment Number]
[Flags | Window Size]
[Checksum | Urgent Pointer]
[Options]

TCP features:
- Connection-oriented (3-way handshake)
- Reliable (retransmission)
- In-order delivery
- Flow control (receiver window)
- Congestion control (sender rate)

TCP congestion control:
1. Slow start: cwnd doubles each RTT
2. Congestion avoidance: cwnd += 1 each RTT
3. On loss: 
   - Timeout: cwnd = 1, slow start
   - 3 dup ACKs: cwnd /= 2, fast recovery
```

```
TCP state diagram (simplified):
CLOSED → [SYN sent] → SYN_SENT
                          ↓ [SYN+ACK received, ACK sent]
                      ESTABLISHED
                          ↓ [FIN sent]
                      FIN_WAIT_1
                          ↓ [ACK received]
                      FIN_WAIT_2
                          ↓ [FIN received, ACK sent]
                      TIME_WAIT
                          ↓ [timeout]
                      CLOSED
```

**Time:** 3 weeks

</details>

<details>
<summary><strong>Chapter 4: Network Layer - Data Plane</strong></summary>

**Topics:**
- Router architecture
- IP addressing
- NAT
- IPv6

**Key Concepts:**
```
IPv4 header:
[Version | IHL | TOS | Total Length]
[Identification | Flags | Fragment Offset]
[TTL | Protocol | Header Checksum]
[Source IP Address]
[Destination IP Address]
[Options]

IP addressing:
- 32 bits (IPv4): 192.168.1.1
- CIDR notation: 192.168.1.0/24 (24-bit network prefix)
- Subnet mask: 255.255.255.0

Special addresses:
- 10.0.0.0/8: Private
- 172.16.0.0/12: Private
- 192.168.0.0/16: Private
- 127.0.0.0/8: Loopback
- 0.0.0.0: This network
- 255.255.255.255: Broadcast

NAT (Network Address Translation):
- Private IP ←→ Public IP:Port
- Allows many devices to share one public IP
- Breaks end-to-end principle
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 5: Network Layer - Control Plane</strong></summary>

**Topics:**
- Routing algorithms
- OSPF
- BGP
- SDN

**Key Concepts:**
```
Link-state routing (OSPF):
1. Each router knows complete topology
2. Dijkstra's algorithm computes shortest paths
3. Flood link-state advertisements

Distance-vector routing (RIP):
1. Each router knows only neighbors
2. Exchange distance vectors with neighbors
3. Bellman-Ford equation: Dx(y) = min{c(x,v) + Dv(y)}

BGP (Border Gateway Protocol):
- Inter-domain routing
- Path-vector protocol
- Policy-based routing
- AS (Autonomous System) level
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 6: Link Layer</strong></summary>

**Topics:**
- Error detection/correction
- Multiple access protocols
- Switched LANs
- Ethernet
- VLANs

**Key Concepts:**
```
MAC address:
- 48 bits, hardware address
- Format: AA:BB:CC:DD:EE:FF
- Unique to each network interface

Ethernet frame:
[Preamble | Dest MAC | Src MAC | Type | Data | CRC]

Multiple access protocols:
- CSMA/CD (Ethernet): Carrier sense, collision detection
- CSMA/CA (WiFi): Collision avoidance (RTS/CTS)

Switch vs Router:
- Switch: Layer 2, uses MAC addresses
- Router: Layer 3, uses IP addresses

ARP (Address Resolution Protocol):
- IP address → MAC address
- Broadcast request, unicast reply
- Cached in ARP table
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 7: Wireless and Mobile Networks</strong></summary>

**Topics:**
- WiFi (802.11)
- Cellular networks
- Mobility management

**Time:** 1 week (can skim)

</details>

<details>
<summary><strong>Chapter 8: Security</strong></summary>

**Topics:**
- Cryptography basics
- Authentication
- Message integrity
- SSL/TLS
- Network layer security
- Firewalls

**Key Concepts:**
```
Symmetric encryption:
- Same key for encrypt/decrypt
- Fast (AES)
- Key distribution problem

Asymmetric encryption:
- Public/private key pair
- Slow (RSA)
- Solves key distribution

TLS handshake (simplified):
1. Client → Server: Client Hello (cipher suites)
2. Server → Client: Server Hello, Certificate
3. Client: Verify certificate, generate session key
4. Client → Server: Encrypted session key (with server's public key)
5. Both: Derive symmetric keys, encrypted communication
```

**Time:** 1-2 weeks

</details>

### How to Study Kurose & Ross

1. **Use Wireshark** - Capture and analyze real packets
2. **Do the labs** - Wireshark labs are excellent
3. **Build applications** - HTTP server, DNS resolver
4. **Run protocol simulations** - Understand TCP behavior
5. **Read RFCs** - For protocols you use

---

## 7. Readings in Database Systems (The Red Book)

### Book Overview

| Detail | Info |
|--------|------|
| Editor | Peter Bailis, Joseph Hellerstein, Michael Stonebraker |
| Format | Collection of classic papers with commentary |
| Difficulty | Advanced |
| Free Online | Yes - redbook.io |

### What It Teaches
Database system design through classic and influential research papers.

### Section Breakdown

<details>
<summary><strong>Section 1: Background</strong></summary>

**Papers:**
- "What Goes Around Comes Around" (Stonebraker & Hellerstein)
  - History of data models
  - IMS, CODASYL, relational, OO, XML

**Key Takeaways:**
- Data models come and go
- Relational model's longevity
- Why certain approaches failed

**Time:** 1 week

</details>

<details>
<summary><strong>Section 2: Traditional RDBMS Systems</strong></summary>

**Papers:**
- "Architecture of a Database System" (Hellerstein et al.)
  - Complete overview of RDBMS architecture
  - Process models, query processing, storage

**Key Concepts:**
```
DBMS Architecture:
                    ┌─────────────────┐
                    │  Client/API     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Query Parser    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Query Optimizer │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Query Executor  │
                    └────────┬────────┘
                             ↓
         ┌──────────┴──────────┬──────────────┐
         ↓                     ↓              ↓
┌─────────────┐     ┌──────────────┐  ┌────────────┐
│ Transaction │     │ Access       │  │ Storage    │
│ Manager     │     │ Methods      │  │ Manager    │
└─────────────┘     └──────────────┘  └────────────┘
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Section 3: Query Processing</strong></summary>

**Papers:**
- "Access Path Selection in a Relational Database Management System" (Selinger et al.)
  - System R optimizer
  - Cost-based optimization

**Key Concepts:**
```
Query optimization:
1. Parse SQL → logical plan
2. Enumerate equivalent plans
3. Estimate cost of each plan
4. Choose cheapest

Cost estimation factors:
- Cardinality (number of rows)
- Selectivity (fraction passing predicate)
- I/O cost (disk accesses)
- CPU cost

Join algorithms:
- Nested loop: O(n × m)
- Sort-merge: O(n log n + m log m)
- Hash join: O(n + m)
```

```sql
-- EXPLAIN shows query plan
EXPLAIN ANALYZE
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'USA';
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Section 4: Transactions</strong></summary>

**Papers:**
- "Concurrency Control and Recovery" (Franklin)
- "ARIES" (Mohan et al.)

**Key Concepts:**
```
ACID:
- Atomicity: All or nothing
- Consistency: Valid state → valid state
- Isolation: Transactions don't interfere
- Durability: Committed = permanent

Concurrency control:
1. Two-phase locking (2PL):
   - Growing phase: Acquire locks
   - Shrinking phase: Release locks
   
2. Timestamp ordering:
   - Each transaction gets timestamp
   - Operations ordered by timestamp

3. MVCC (Multi-Version Concurrency Control):
   - Readers don't block writers
   - Writers don't block readers
   - Multiple versions of data

ARIES recovery:
1. Analysis: Identify dirty pages, active transactions
2. Redo: Replay history to crash point
3. Undo: Roll back uncommitted transactions
```

**Time:** 2-3 weeks

</details>

<details>
<summary><strong>Section 5: Distributed Databases</strong></summary>

**Papers:**
- "CAP Twelve Years Later" (Brewer)
- "Spanner: Google's Globally-Distributed Database"

**Key Concepts:**
```
CAP theorem:
- Consistency: All nodes see same data
- Availability: Every request gets response
- Partition tolerance: System works despite network partitions

During partition, choose:
- CP: Consistent but some requests fail
- AP: Available but possibly stale data

Distributed commit:
- 2PC (Two-phase commit): Blocking, single coordinator
- 3PC: Non-blocking but complex
- Paxos/Raft: Consensus with leader election
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Section 6-11: Modern Topics</strong></summary>

**Sections:**
- Column stores
- Data integration
- MapReduce and beyond
- Machine learning and databases
- Query optimization revisited
- Web data

**Time:** 2-3 weeks (selective reading)

</details>

### Alternative: Database Management Systems (Ramakrishnan)

More textbook-style, better for learning implementation:

| Chapter | Topic |
|---------|-------|
| 1-5 | Relational model, SQL |
| 6-7 | Database design, normalization |
| 8-12 | Storage, indexing |
| 13-17 | Query processing, optimization |
| 18-20 | Transaction management |

### How to Study Databases

1. **Read "Architecture of a Database System"** first
2. **Implement a simple database** (SQL parser, B-tree, buffer pool)
3. **Study PostgreSQL source** - well-documented
4. **Write complex SQL queries** - understand execution
5. **Experiment with different databases** - Compare behavior

---

## 8. Crafting Interpreters

### Book Overview

| Detail | Info |
|--------|------|
| Author | Robert Nystrom |
| Pages | ~800 |
| Languages | Java (Part II), C (Part III) |
| Difficulty | Moderate to Challenging |
| Free Online | Yes - craftinginterpreters.com |

### What It Teaches
Build two complete interpreters for the Lox language: a tree-walking interpreter in Java and a bytecode VM in C.

### Part I: Welcome

<details>
<summary><strong>Chapters 1-3: Introduction</strong></summary>

**Topics:**
- What interpreters and compilers do
- The Lox language
- Map of the book

**Lox language features:**
```javascript
// Variables
var name = "Lox";

// Functions
fun greet(name) {
  print "Hello, " + name + "!";
}

// Classes
class Animal {
  init(name) {
    this.name = name;
  }
  
  speak() {
    print this.name + " makes a sound";
  }
}

class Dog < Animal {
  speak() {
    print this.name + " barks";
  }
}
```

**Time:** 2-3 days

</details>

### Part II: A Tree-Walk Interpreter (Java)

<details>
<summary><strong>Chapter 4: Scanning (Lexer)</strong></summary>

**Topics:**
- Lexemes and tokens
- Regular languages
- Scanner implementation

**Key Concepts:**
```java
// Token types
enum TokenType {
  // Single-character tokens
  LEFT_PAREN, RIGHT_PAREN, LEFT_BRACE, RIGHT_BRACE,
  COMMA, DOT, MINUS, PLUS, SEMICOLON, SLASH, STAR,
  
  // Literals
  IDENTIFIER, STRING, NUMBER,
  
  // Keywords
  AND, CLASS, ELSE, FALSE, FUN, FOR, IF, NIL, OR,
  PRINT, RETURN, SUPER, THIS, TRUE, VAR, WHILE,
  
  EOF
}

// Scanner structure
class Scanner {
  private final String source;
  private final List<Token> tokens = new ArrayList<>();
  private int start = 0;    // Start of current lexeme
  private int current = 0;  // Current character
  private int line = 1;
  
  List<Token> scanTokens() {
    while (!isAtEnd()) {
      start = current;
      scanToken();
    }
    tokens.add(new Token(EOF, "", null, line));
    return tokens;
  }
  
  private void scanToken() {
    char c = advance();
    switch (c) {
      case '(': addToken(LEFT_PAREN); break;
      case ')': addToken(RIGHT_PAREN); break;
      // ... more cases
      case '"': string(); break;
      default:
        if (isDigit(c)) number();
        else if (isAlpha(c)) identifier();
        else error(line, "Unexpected character.");
    }
  }
}
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 5-6: Representing and Parsing Expressions</strong></summary>

**Topics:**
- AST (Abstract Syntax Tree)
- Visitor pattern
- Recursive descent parsing
- Precedence and associativity

**Key Concepts:**
```
Expression grammar (precedence low to high):
expression → equality
equality   → comparison (("!=" | "==") comparison)*
comparison → term ((">" | ">=" | "<" | "<=") term)*
term       → factor (("-" | "+") factor)*
factor     → unary (("/" | "*") unary)*
unary      → ("!" | "-") unary | primary
primary    → NUMBER | STRING | "true" | "false" | "nil"
           | "(" expression ")"
```

```java
// AST nodes
abstract class Expr {
  abstract <R> R accept(Visitor<R> visitor);
}

class Binary extends Expr {
  final Expr left;
  final Token operator;
  final Expr right;
  
  Binary(Expr left, Token operator, Expr right) {
    this.left = left;
    this.operator = operator;
    this.right = right;
  }
  
  <R> R accept(Visitor<R> visitor) {
    return visitor.visitBinaryExpr(this);
  }
}

// Recursive descent parser
class Parser {
  private Expr expression() {
    return equality();
  }
  
  private Expr equality() {
    Expr expr = comparison();
    while (match(BANG_EQUAL, EQUAL_EQUAL)) {
      Token operator = previous();
      Expr right = comparison();
      expr = new Expr.Binary(expr, operator, right);
    }
    return expr;
  }
  
  // Similar for comparison, term, factor, unary, primary
}
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 7-8: Evaluating and Statements</strong></summary>

**Topics:**
- Tree-walking interpreter
- Statements and state
- Print statements
- Expression statements
- Variable declarations

**Key Concepts:**
```java
// Interpreter using visitor pattern
class Interpreter implements Expr.Visitor<Object>, 
                             Stmt.Visitor<Void> {
  
  private Environment environment = new Environment();
  
  @Override
  public Object visitBinaryExpr(Expr.Binary expr) {
    Object left = evaluate(expr.left);
    Object right = evaluate(expr.right);
    
    switch (expr.operator.type) {
      case PLUS:
        if (left instanceof Double && right instanceof Double) {
          return (double)left + (double)right;
        }
        if (left instanceof String && right instanceof String) {
          return (String)left + (String)right;
        }
        throw new RuntimeError(expr.operator, 
          "Operands must be two numbers or two strings.");
      case MINUS:
        return (double)left - (double)right;
      // ... more cases
    }
  }
  
  @Override
  public Void visitPrintStmt(Stmt.Print stmt) {
    Object value = evaluate(stmt.expression);
    System.out.println(stringify(value));
    return null;
  }
}
```

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapters 9-10: Control Flow and Functions</strong></summary>

**Topics:**
- If statements
- While and for loops
- Function declarations
- Function calls
- Closures

**Key Concepts:**
```java
// Environment (variable storage)
class Environment {
  final Environment enclosing;
  private final Map<String, Object> values = new HashMap<>();
  
  Environment() {
    enclosing = null;
  }
  
  Environment(Environment enclosing) {
    this.enclosing = enclosing;
  }
  
  void define(String name, Object value) {
    values.put(name, value);
  }
  
  Object get(Token name) {
    if (values.containsKey(name.lexeme)) {
      return values.get(name.lexeme);
    }
    if (enclosing != null) return enclosing.get(name);
    throw new RuntimeError(name, 
      "Undefined variable '" + name.lexeme + "'.");
  }
}

// Function implementation
class LoxFunction implements LoxCallable {
  private final Stmt.Function declaration;
  private final Environment closure;
  
  @Override
  public Object call(Interpreter interpreter, 
                     List<Object> arguments) {
    Environment environment = new Environment(closure);
    for (int i = 0; i < declaration.params.size(); i++) {
      environment.define(declaration.params.get(i).lexeme,
                        arguments.get(i));
    }
    
    try {
      interpreter.executeBlock(declaration.body, environment);
    } catch (Return returnValue) {
      return returnValue.value;
    }
    return null;
  }
}
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapters 11-13: Resolution, Classes, Inheritance</strong></summary>

**Topics:**
- Variable resolution
- Class declarations
- Instances and this
- Inheritance and super

**Key Concepts:**
```java
// Resolver (static analysis before interpretation)
class Resolver implements Expr.Visitor<Void>, Stmt.Visitor<Void> {
  private final Interpreter interpreter;
  private final Stack<Map<String, Boolean>> scopes = new Stack<>();
  
  private void resolve(Expr expr) {
    expr.accept(this);
  }
  
  private void resolveLocal(Expr expr, Token name) {
    for (int i = scopes.size() - 1; i >= 0; i--) {
      if (scopes.get(i).containsKey(name.lexeme)) {
        interpreter.resolve(expr, scopes.size() - 1 - i);
        return;
      }
    }
  }
}

// Class implementation
class LoxClass implements LoxCallable {
  final String name;
  final LoxClass superclass;
  private final Map<String, LoxFunction> methods;
  
  LoxFunction findMethod(String name) {
    if (methods.containsKey(name)) {
      return methods.get(name);
    }
    if (superclass != null) {
      return superclass.findMethod(name);
    }
    return null;
  }
}
```

**Time:** 2-3 weeks

</details>

### Part III: A Bytecode Virtual Machine (C)

<details>
<summary><strong>Chapters 14-30: VM Implementation</strong></summary>

**Topics:**
- Chunks of bytecode
- Virtual machine
- Scanning/compiling on the fly
- Types of values
- Strings
- Hash tables
- Global variables
- Local variables
- Control flow
- Functions and calls
- Closures
- Garbage collection
- Classes and instances
- Methods and initializers
- Superclasses
- Optimization

**Key Concepts:**
```c
// Bytecode operations
typedef enum {
  OP_CONSTANT,
  OP_NIL,
  OP_TRUE,
  OP_FALSE,
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_DIVIDE,
  OP_NEGATE,
  OP_RETURN,
  // ... more opcodes
} OpCode;

// Virtual machine
typedef struct {
  Chunk* chunk;
  uint8_t* ip;  // Instruction pointer
  Value stack[STACK_MAX];
  Value* stackTop;
  // ...
} VM;

// Main interpret loop
static InterpretResult run() {
  for (;;) {
    uint8_t instruction;
    switch (instruction = READ_BYTE()) {
      case OP_CONSTANT: {
        Value constant = READ_CONSTANT();
        push(constant);
        break;
      }
      case OP_ADD: BINARY_OP(+); break;
      case OP_SUBTRACT: BINARY_OP(-); break;
      case OP_RETURN: {
        printValue(pop());
        return INTERPRET_OK;
      }
    }
  }
}
```

**Time:** 8-12 weeks

</details>

### How to Study Crafting Interpreters

1. **Type all the code** - Don't copy-paste
2. **Do the challenges** - At end of each chapter
3. **Test incrementally** - Run after each addition
4. **Debug with printf** - Print AST, tokens, bytecode
5. **Extend the language** - Add features after finishing

### Project Progression

```
Part II (Tree-walk):
1. Scanner (tokens)
2. Parser (AST)
3. Evaluator (basic expressions)
4. Statements and variables
5. Control flow
6. Functions
7. Classes

Part III (Bytecode VM):
1. Bytecode representation
2. VM execution
3. Single-pass compiler
4. Variables (global and local)
5. Control flow
6. Functions
7. Closures
8. Garbage collection
9. Classes
10. Optimization
```

---

## 9. Designing Data-Intensive Applications (DDIA)

### Book Overview

| Detail | Info |
|--------|------|
| Author | Martin Kleppmann |
| Pages | ~600 |
| Difficulty | Intermediate to Advanced |
| Year | 2017 (still highly relevant) |

### What It Teaches
How to build reliable, scalable, maintainable data systems. The most important book for backend/infrastructure engineers.

### Part I: Foundations of Data Systems

<details>
<summary><strong>Chapter 1: Reliable, Scalable, and Maintainable Applications</strong></summary>

**Topics:**
- Reliability (faults vs failures)
- Scalability (load and performance)
- Maintainability (operability, simplicity, evolvability)

**Key Concepts:**
```
Reliability:
- Hardware faults: Disk, RAM, network failures
- Software faults: Bugs, cascading failures
- Human errors: Configuration mistakes

Scalability:
- Describing load: Requests/sec, read/write ratio
- Describing performance: Latency percentiles (p50, p99)
- Approaches: Scale up vs scale out

Percentiles:
p50 = median (half faster, half slower)
p99 = 99th percentile (1% slower than this)
p999 = slowest 0.1%

Tail latencies matter for user experience!
```

**Time:** 3-4 days

</details>

<details>
<summary><strong>Chapter 2: Data Models and Query Languages</strong></summary>

**Topics:**
- Relational model
- Document model
- Graph model
- Query languages (SQL, MapReduce, Cypher)

**Key Concepts:**
```
Relational (SQL):
- Tables with rows and columns
- Schema enforced
- Joins for relationships
- Good for: Complex queries, many-to-many relationships

Document (MongoDB):
- Nested JSON-like documents
- Schema flexible
- Good for: Self-contained documents, one-to-many relationships

Graph (Neo4j):
- Nodes and edges
- Good for: Highly connected data, many-to-many relationships

When to use what:
- Relational: Most applications, complex queries
- Document: Content management, real-time analytics
- Graph: Social networks, recommendation engines
```

**Time:** 1 week

</details>

<details>
<summary><strong>Chapter 3: Storage and Retrieval</strong></summary>

**Topics:**
- Log-structured storage (LSM trees)
- Page-oriented storage (B-trees)
- OLTP vs OLAP
- Column-oriented storage

**Key Concepts:**
```
B-tree:
- Balanced tree, pages on disk
- In-place updates
- O(log n) reads and writes
- Good for: Random reads, range scans

LSM-tree (Log-Structured Merge-tree):
- Append-only writes to memory (memtable)
- Periodic compaction to disk (SSTables)
- Good for: Write-heavy workloads

Write Amplification:
B-tree: Write page for every update
LSM: Multiple writes during compaction

Read Amplification:
B-tree: O(log n) pages
LSM: Check multiple SSTables

OLTP vs OLAP:
OLTP: Many small transactions (web apps)
OLAP: Few large analytical queries (data warehouse)

Column storage:
- Store each column separately
- Compress well (same data type)
- Great for analytics (aggregate columns)
```

```
LSM-tree structure:
                ┌──────────────┐
Write ───────→  │   Memtable   │  (in-memory)
                └──────┬───────┘
                       ↓ (flush when full)
                ┌──────────────┐
                │  SSTable L0  │  (on disk, sorted)
                └──────┬───────┘
                       ↓ (compact)
                ┌──────────────┐
                │  SSTable L1  │
                └──────┬───────┘
                       ↓
                      ...
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 4: Encoding and Evolution</strong></summary>

**Topics:**
- Language-specific formats
- JSON, XML, Binary formats
- Thrift, Protocol Buffers, Avro
- Schema evolution

**Key Concepts:**
```
Data encoding:
- JSON/XML: Human-readable, verbose
- Protocol Buffers/Thrift: Binary, schema required
- Avro: Binary, schema in file

Schema evolution:
- Forward compatibility: Old code reads new data
- Backward compatibility: New code reads old data

Strategies:
- Add optional fields (backward compatible)
- Never remove required fields
- Never reuse field numbers
```

**Time:** 1 week

</details>

### Part II: Distributed Data

<details>
<summary><strong>Chapter 5: Replication</strong></summary>

**Topics:**
- Leaders and followers
- Replication lag
- Multi-leader replication
- Leaderless replication

**Key Concepts:**
```
Single-leader replication:
- One leader accepts writes
- Followers replicate from leader
- Read from followers (eventual consistency)

Replication lag problems:
- Reading your writes: User sees old data after write
- Monotonic reads: User sees time go backward
- Consistent prefix reads: Causality violated

Multi-leader replication:
- Multiple data centers, each with leader
- Conflict resolution needed
- Use cases: Multi-datacenter, offline clients

Leaderless replication (Dynamo-style):
- Write to multiple nodes
- Read from multiple nodes
- Quorum: W + R > N for consistency
- Example: W=2, R=2, N=3

Conflict resolution:
- Last write wins (LWW)
- Merge values
- Custom resolution
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 6: Partitioning</strong></summary>

**Topics:**
- Partitioning strategies
- Secondary indexes
- Rebalancing
- Request routing

**Key Concepts:**
```
Partitioning strategies:
1. Key range: [a-f], [g-m], [n-z]
   - Good for range queries
   - Risk of hot spots
   
2. Hash: hash(key) mod N
   - Even distribution
   - No range queries

Secondary indexes:
- Local index: Each partition indexes its data
- Global index: Partition index separately from data

Rebalancing:
- Fixed partitions: Over-provision, move whole partitions
- Dynamic: Split/merge based on size
- Fixed per node: # partitions = # nodes × constant

Service discovery:
- Client contacts any node → routes to correct
- Routing tier (proxy)
- Client knows partition assignment
```

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 7: Transactions</strong></summary>

**Topics:**
- ACID properties
- Single-object vs multi-object
- Weak isolation levels
- Serializability

**Key Concepts:**
```
Isolation levels:

Read Committed:
- No dirty reads (only see committed data)
- No dirty writes (don't overwrite uncommitted)

Snapshot Isolation (MVCC):
- Each transaction sees consistent snapshot
- Writers don't block readers
- Implementation: Multi-version, each row has transaction ID

Serializable:
- Transactions execute as if serial
- Implementations:
  1. Actual serial execution (single thread)
  2. Two-phase locking (2PL)
  3. Serializable snapshot isolation (SSI)

Two-phase locking:
- Shared locks for reads
- Exclusive locks for writes
- Hold all locks until commit
- Can deadlock

SSI:
- Optimistic: Allow transactions to proceed
- Check for conflicts at commit
- Abort if conflict detected
```

**Time:** 2 weeks

</details>

<details>
<summary><strong>Chapter 8: The Trouble with Distributed Systems</strong></summary>

**Topics:**
- Unreliable networks
- Unreliable clocks
- Knowledge, truth, and lies

**Key Concepts:**
```
Network problems:
- Request lost
- Request queued
- Remote node crashed
- Response lost
- Response delayed

You can't tell which happened!

Timeout:
- Too short: False positives (node seems dead but isn't)
- Too long: Long wait for actual failures

Clock issues:
- Time-of-day clocks: Can jump backward (NTP sync)
- Monotonic clocks: Only for measuring duration

Ordering events:
- Happened-before relationship
- Lamport timestamps: Logical ordering
- Vector clocks: Capture causality

Byzantine faults:
- Nodes may lie or behave maliciously
- Need 3f+1 nodes to tolerate f Byzantine faults
```

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 9: Consistency and Consensus</strong></summary>

**Topics:**
- Linearizability
- Ordering guarantees
- Distributed transactions
- Consensus

**Key Concepts:**
```
Linearizability:
- Strongest consistency model
- Operations appear atomic, instantaneous
- Total order of operations
- Expensive: Coordination required

CAP theorem:
- During network partition, choose:
  - Consistency (linearizability): Reject some requests
  - Availability: Serve requests, maybe stale

Consensus:
- Agreement: All nodes decide same value
- Validity: Value was proposed by some node
- Termination: Eventually decides
- Integrity: Decide exactly once

Algorithms:
- Paxos: Classic, complex
- Raft: Understandable, leader-based
- Zab: ZooKeeper's algorithm

Use cases for consensus:
- Leader election
- Atomic commit
- Total order broadcast
```

**Time:** 2-3 weeks

</details>

### Part III: Derived Data

<details>
<summary><strong>Chapter 10: Batch Processing</strong></summary>

**Topics:**
- Unix tools philosophy
- MapReduce
- Beyond MapReduce

**Key Concepts:**
```
MapReduce:
1. Map: (key, value) → [(k2, v2), ...]
2. Shuffle: Group by key
3. Reduce: (key, [values]) → (key, result)

Example - Word count:
Map: "hello world" → [("hello", 1), ("world", 1)]
Reduce: ("hello", [1, 1, 1]) → ("hello", 3)

Joins in MapReduce:
- Sort-merge join: Sort both sides by join key
- Broadcast hash join: Small table fits in memory
- Partitioned hash join: Partition both sides same way

Beyond MapReduce:
- Spark: Keep data in memory between stages
- Flink: Stream processing with batch
- Dataflow model: Unified batch and stream
```

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 11: Stream Processing</strong></summary>

**Topics:**
- Message brokers
- Stream processing concepts
- Event time vs processing time

**Key Concepts:**
```
Messaging systems:
- Direct messaging (UDP, HTTP)
- Message brokers (RabbitMQ)
- Log-based (Kafka)

Kafka:
- Append-only log per partition
- Consumers track their position (offset)
- Replay possible
- Retention based on time or size

Stream processing patterns:
- Event-at-a-time
- Micro-batching
- Windowing (tumbling, sliding, session)

Time:
- Event time: When event occurred
- Processing time: When event processed
- Watermarks: "No more events before time T"

Exactly-once semantics:
- Idempotent operations
- Transactional writes
- Deduplication
```

**Time:** 1-2 weeks

</details>

<details>
<summary><strong>Chapter 12: The Future of Data Systems</strong></summary>

**Topics:**
- Data integration
- Unbundling databases
- Correctness
- Ethics

**Key Concepts:**
```
Data integration:
- CDC (Change Data Capture): Stream changes from DB
- Event sourcing: Store events, not current state
- CQRS: Separate read and write models

Total order broadcast + Consensus:
- Log as single source of truth
- Derived views for different queries
- Rebuild views from log

Lambda architecture:
- Batch layer: Complete, accurate, slow
- Speed layer: Approximate, fast
- Serving layer: Combine results

Kappa architecture:
- Stream processing only
- Reprocess by replaying log
```

**Time:** 1 week

</details>

### How to Study DDIA

1. **Read cover to cover** - It builds on itself
2. **Take notes** - The diagrams are important
3. **Relate to systems you use** - PostgreSQL, Redis, Kafka
4. **Read the references** - Papers go deeper
5. **Build something** - Implement a simple distributed system

---

## Summary: Study Order and Time Estimates

### Recommended Order

| Order | Book | Weeks |
|-------|------|-------|
| 1 | SICP (or HtDP) | 12-16 |
| 2 | Algorithm Design Manual | 10-14 |
| 3 | Math for CS (6.042) | 10-12 |
| 4 | CS:APP | 14-18 |
| 5 | OSTEP | 10-14 |
| 6 | Networking (Kurose) | 10-12 |
| 7 | Crafting Interpreters | 14-20 |
| 8 | Database (Red Book or Ramakrishnan) | 10-14 |
| 9 | DDIA | 10-12 |

**Total: ~100-130 weeks (2-2.5 years part-time)**

### Parallel Study Option

```
Track A (Foundations):        Track B (Systems):
Week 1-12: SICP               Week 1-12: CS:APP
Week 13-24: Algorithms        Week 13-24: OSTEP  
Week 25-36: Math              Week 25-36: Networking

Then:
Week 37-52: Crafting Interpreters
Week 53-64: Databases
Week 65-76: DDIA
```

### Quick Reference: Key Chapters

| Book | Must-Read Chapters |
|------|-------------------|
| SICP | 1, 2, 4 |
| Algorithm Design Manual | 1-10 |
| Math 6.042 | 1-5 (Proofs), 11-14 (Counting/Probability) |
| CS:APP | 2, 3, 5, 6, 8, 9 |
| OSTEP | Virtualization (1-24), Concurrency (25-34) |
| Kurose | 1-5 |
| Crafting Interpreters | All (build both interpreters) |
| Red Book | Architecture paper, Query Processing |
| DDIA | All |