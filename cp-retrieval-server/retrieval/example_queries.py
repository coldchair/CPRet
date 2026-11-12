"""Example queries for the search box.

Each language has problems of increasing difficulty; every problem offers three query styles:
the solution idea, a short statement, and a full statement rewritten in different words.
All targets rank on the first page of results (checked against the 2026-11 database).
"""

EXAMPLE_QUERIES = {
    "zh": [
        {
            "level": "easy",
            "title": "[CSP-J 2023] 公路",
            "topic": "贪心",
            "url": "https://www.luogu.com.cn/problem/P9749",
            "idea": "贪心：沿途维护目前遇到的最低油价，在便宜的站点买油一直用到遇见更便宜的站点为止；油只能按整数升购买，多出来的油留着继续用。",
            "short": "一条公路上有 n 个加油站，相邻两站距离为 v_i，第 i 站油价为 a_i，每升油能跑 d 公里，只能买整数升，油箱无限大，初始没油。求从第 1 站开到第 n 站的最少花费。",
            "full": "快递员要骑电动车沿一条直路从 1 号驿站送货到 n 号驿站，沿途依次经过编号 1 到 n 的驿站，i 号驿站和 i+1 号驿站之间相距 v_i 公里。\n\n每个驿站都能给电动车换电，在 i 号驿站每份电量的价格是 a_i 元，每份电量可以让车行驶 d 公里，并且只能购买整数份。电池容量可以视为无限大，出发时电量为 0。\n\n求快递员从 1 号驿站到达 n 号驿站最少需要花多少钱。\n\n输入格式：第一行两个正整数 n, d；第二行 n-1 个正整数 v_1..v_{n-1}；第三行 n 个正整数 a_1..a_n。\n输出格式：一个整数，表示最少花费。\n\n样例输入：\n5 4\n10 10 10 10\n9 8 9 6 5\n样例输出：\n79\n\n数据范围：1 ≤ n ≤ 10^5，1 ≤ d, v_i, a_i ≤ 10^5。",
        },
        {
            "level": "medium",
            "title": "[CSP-J 2023] 旅游巴士",
            "topic": "分层图最短路",
            "url": "https://www.luogu.com.cn/problem/P9751",
            "idea": "分层图最短路：把状态设为（点，到达时间 mod k），用 Dijkstra 求每个状态的最早到达时间；如果边的开放时间还没到，就把出发时间整体推迟 k 的整数倍来等待，最后取终点上时间模 k 为 0 的状态。",
            "short": "有向图有 n 个点 m 条边，每条边走过去恰好花 1 单位时间，第 i 条边只有在时刻 a_i 之后才能通过。只能在 k 的倍数时刻从 1 号点出发，并在 k 的倍数时刻到达 n 号点，途中不能停留。求最早的到达时刻，无解输出 -1。",
            "full": "仓库里有 n 个货位和 m 条单向传送带，1 号货位是装货口，n 号货位是卸货口。从 0 时刻起，每隔 k 秒装货口会放出一个机器人，卸货口也只在 k 的整数倍时刻收货。\n\n机器人沿任意一条传送带移动都恰好用 1 秒，而且机器人一旦出发就不能在任何货位或传送带上停下等待。第 i 条传送带从货位 u_i 通向货位 v_i，并且只有在时刻不早于 a_i 时才启动，之前不能通过。\n\n请选择机器人的出发时刻和路线，使它到达卸货口并被收货的时刻尽量早。如果无法做到，输出 -1。\n\n输入：第一行 n m k；接下来 m 行，每行 u_i v_i a_i。\n输出：最早的收货时刻。\n\n样例输入：\n5 5 3\n1 2 0\n2 5 1\n1 3 0\n3 4 3\n4 5 1\n样例输出：\n6\n\n数据范围：2 ≤ n ≤ 10^4，1 ≤ m ≤ 2×10^4，1 ≤ k ≤ 100，0 ≤ a_i ≤ 10^6。",
        },
        {
            "level": "hard",
            "title": "[CSP-S 2023] 消消乐",
            "topic": "栈 + 哈希计数",
            "url": "https://www.luogu.com.cn/problem/P9753",
            "idea": "像括号匹配一样用栈做相邻相同字符的消除：对每个前缀记录消除后剩下的栈状态，两个前缀的栈状态相同时，它们之间的子串就能被完全消除；用哈希或 Trie 给栈状态编号，统计相同状态的前缀对数。",
            "short": "给定一个长度为 n 的小写字母串，每次操作可以删除两个相邻且相同的字符，剩下部分拼接起来。问有多少个非空连续子串可以通过若干次操作被删成空串。n ≤ 2×10^6。",
            "full": "小 A 在玩一个一维的消除游戏。给定一个长度为 n、只包含小写字母的字符串，每次操作可以删除字符串中两个相邻且相同的字符，删除后剩余部分自动拼接。如果一个字符串能通过若干次操作变成空串，就称它是可消除的。\n\n例如 abba 可以先删去 bb 再删去 aa，因此是可消除的；而 abab 不是。\n\n请求出给定字符串的所有非空连续子串中，可消除的子串有多少个（位置不同的相同子串分别计数）。\n\n输入：第一行一个整数 n；第二行一个长度为 n 的小写字母串。\n输出：一个整数，表示可消除的连续子串个数。\n\n样例输入：\n8\naccabccb\n样例输出：\n5\n\n数据范围：1 ≤ n ≤ 2×10^6，需要线性或接近线性的做法。",
        },
    ],
    "en": [
        {
            "level": "easy",
            "title": "[ABC340D] Super Takahashi Bros.",
            "topic": "Dijkstra",
            "url": "https://atcoder.jp/contests/abc340/tasks/abc340_d",
            "idea": "Model the levels as a graph: each node i has an edge to i+1 with cost A_i and an edge to X_i with cost B_i (which may jump backwards), then run Dijkstra from node 1 to node N.",
            "short": "There are N stages. At stage i you can spend A_i seconds to unlock stage i+1, or spend B_i seconds to unlock stage X_i. Starting at stage 1, find the minimum time needed to be able to play stage N.",
            "full": "A game consists of N levels numbered 1 to N. Initially only level 1 is available.\n\nFor each level i (1 <= i <= N-1), once it is available you can clear it in one of two ways:\n- spend A_i seconds, which makes level i+1 available;\n- spend B_i seconds, which makes level X_i available.\n\nIgnoring everything except the time spent clearing levels, what is the minimum number of seconds needed before level N becomes available?\n\nInput: the first line contains N. Each of the next N-1 lines contains A_i, B_i and X_i.\nOutput: print the answer.\n\nConstraints: 2 <= N <= 2*10^5, 1 <= A_i, B_i <= 10^9, 1 <= X_i <= N. All input values are integers.\n\nSample input:\n5\n100 200 3\n50 10 1\n100 200 5\n150 1 2\nSample output:\n350",
        },
        {
            "level": "medium",
            "title": "[ABC348E] Minimize Sum of Distances",
            "topic": "rerooting DP",
            "url": "https://atcoder.jp/contests/abc348/tasks/abc348_e",
            "idea": "Rerooting DP on a tree: one DFS computes f(root) and the total weight of every subtree, then moving the root from a parent p to a child c gives f(c) = f(p) + W - 2 * sub(c), where W is the total weight; take the minimum over all vertices.",
            "short": "Given a tree with N vertices where vertex i has a positive weight C_i, find the minimum over all vertices v of sum_i C_i * dist(v, i), where dist is the number of edges on the path.",
            "full": "A kingdom has N towns connected by N-1 roads so that every pair of towns is connected by exactly one route. Town i needs C_i tons of supplies every day.\n\nThe king wants to build one warehouse in some town v. Delivering one ton of supplies along one road costs 1 coin, so supplying town i from the warehouse costs C_i multiplied by the number of roads between v and i.\n\nChoose the town for the warehouse so that the total daily delivery cost is minimized, and print this minimum cost.\n\nInput: N, then N-1 lines with the endpoints A_i B_i of each road, then C_1 ... C_N.\nOutput: the minimum total cost.\n\nSample input:\n4\n1 2\n1 3\n2 4\n1 1 1 2\nSample output:\n5\n\nConstraints: 1 <= N <= 10^5, 1 <= C_i <= 10^9.",
        },
        {
            "level": "hard",
            "title": "[ABC339G] Smaller Sum",
            "topic": "merge sort tree",
            "url": "https://atcoder.jp/contests/abc339/tasks/abc339_g",
            "idea": "Answer range queries 'sum of the elements not greater than X in A[L..R]' online: build a merge sort tree where each segment tree node stores its values sorted with prefix sums, and binary search in each covering node for O(log^2 N) per query. Queries are XOR-encrypted with the previous answer, so offline sorting does not work.",
            "short": "Given an array A of length N, answer Q queries online. Each query (L, R, X) asks for the sum of A_i over L <= i <= R with A_i <= X. Every query is encoded by XOR with the previous answer. N, Q <= 2*10^5.",
            "full": "A shop records the prices of N items placed on a shelf from left to right, where the i-th item costs A_i.\n\nCustomers arrive one by one. Each customer has a budget X and looks only at the items in positions L to R. They take every item in that range whose price is at most X, and the shop wants to know the total price of the items taken (items are restocked immediately, so the shelf never changes).\n\nTo prevent precomputation, the queries are given encrypted: you receive alpha, beta, gamma, and the real query is L = alpha xor B, R = beta xor B, X = gamma xor B, where B is the answer to the previous customer (B = 0 for the first one). You must answer each customer before reading the next one.\n\nInput: N, then A_1 ... A_N, then Q, then Q lines alpha_i beta_i gamma_i.\nOutput: Q lines, the total price for each customer.\n\nSample input:\n8\n2 0 2 4 0 2 0 3\n5\n1 8 3\n10 12 11\n3 3 2\n3 6 5\n12 0 11\nSample output:\n9\n2\n0\n8\n5\n\nConstraints: 1 <= N, Q <= 2*10^5, 0 <= A_i <= 10^9.",
        },
    ],
}
