
# 🔍 Test Cases of CPRet

## 📌 System Launch Overview

* Our open-source **problem retrieval platform** was launched on **May 21, 2025**: http://1.94.255.218:5000/
* In less than a week, the platform has recorded nearly **2,000 search queries**.
* The blog post introducing the system on **Codeforces** has received over **250 upvotes** and positive community feedback: https://codeforces.com/blog/entry/143098
* **Use cases** include:

  * **Similar Problem Retrieval**: Assisting contestants in expanding their problem-solving perspective and addressing knowledge blind spots.
  * **Duplicate Problem Retrieval**: Helping problem setters identify previously seen ideas or solutions early on.

---

## 🧪 Test Case 1: Duplicate Retrieval in a Recent Contest

**Target Contest**

* **Name**: The 2025 CCPC National Invitational Contest (Northeast), The 19th Northeast Collegiate Programming Contest
* **Url**: https://codeforces.com/gym/105924
* **Date**: May 25, 2025
* **Total Problems**: 12

**Key Findings**

* The system successfully identified **6 problems** with highly similar or identical historical counterparts.
* Only the **top 3 results per query** were manually inspected, indicating a **duplicate rate of at least 50%**—likely higher in practice.
* Under current scoring rules, solving these 6 problems is enough to secure a **silver medal**, and just one additional easy problem would result in a **gold medal**, raising **serious fairness concerns**.

**Detection Summary**

| Contest Problem                                                          | Matched Historical Problem                                                       | Similarity Level | Rank |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ---------------- | ---- |
| [A. GD Ultimate Rhythm Lab](https://codeforces.com/gym/105924/problem/A) | [Nowcoder - 小睿睿的数列](https://ac.nowcoder.com/acm/problem/24479)                   | Same approach    | 1    |
| [D. Defend the Carrot](https://codeforces.com/gym/105924/problem/D)      | [SPOJ - UOFTBB](https://www.spoj.com/problems/UOFTBB/)                           | Almost identical | 1    |
| [E. Tree Edge Removal](https://codeforces.com/gym/105924/problem/E)      | [Luogu - \[JRKSJ R7\] 茎](https://www.luogu.com.cn/problem/P8935)                 | Almost identical | 1    |
| [F. Youthful Oath II](https://codeforces.com/gym/105924/problem/F)       | [Codeforces - 80B Depression](https://codeforces.com/problemset/problem/80/B)    | Almost identical | 1    |
| [J. Kingdom: Memories](https://codeforces.com/gym/105924/problem/J)      | [AtCoder - R Walk](https://atcoder.jp/contests/dp/tasks/dp_r)                    | Almost identical | 3    |
| [L. Bathhouse](https://codeforces.com/gym/105924/problem/L)              | [Codeforces - 219E Parking Lot](https://codeforces.com/problemset/problem/219/E) | Same approach    | 2    |

---

## 🧪 Test Case 2: Similar Problem Retrieval – MEX Variants

We conducted a query with the classic problem "**interval MEX**" to identify its **variants across different contests**, aiming to showcase the system’s utility for **idea expansion and knowledge transfer**.

### Query Problem Description

> Given a sequence of $n$ natural numbers $a[1..n]$, answer $m$ queries.
> Each query specifies a range $[l, r]$, and asks for $\mathrm{mex}({a_l, a_{l+1}, \dots, a_r})$ —
> the **minimum excluded value** in the subarray.
>
> This problem can be solved in **$O((n + m) log n)$** time using segment trees.

### Retrieval Results

| Rank | Title                                                                                          | Description                                                            |
| ---- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 1    | [Luogu P4137: RMQ Problem / mex](https://www.luogu.com.cn/problem/P4137)                       | Original problem                                                       |
| 2    | [LOJ 6908: THUPC 2024 Prelim - "Matryoshka"](https://loj.ac/p/6908)                            | MEX of all subarrays of length $k$, then take the MEX of those       |
| 5    | [AtCoder ABC194E: Mex Min](https://atcoder.jp/contests/abc194/tasks/abc194_e)                  | MEX of all subarrays of length $k$, then take the **minimum**        |
| 6    | [Luogu P10032: Mex of Sequence](https://www.luogu.com.cn/problem/P10032)                       | Repeated operations: $a'[i] = \mathrm{mex}(a \setminus a[i])$      |
| 11   | [Nowcoder 237670: “经典”问题](https://ac.nowcoder.com/acm/problem/237670)                          | MEX queries on permutations, optimized to $O(n + m)$                 |
| 14   | [Luogu P8087: 『JROI-5』Interval](https://www.luogu.com.cn/problem/P8087)                        | MEX of all subarrays of length $k$, then take the **maximum**        |
| 15   | [AtCoder ABC290C: Max MEX](https://atcoder.jp/contests/abc290/tasks/abc290_c)                  | MEX of all **subsequences** of length $k$, then take the **minimum** |
| 16   | [Codeforces 1436E: Complicated Computations](https://codeforces.com/problemset/problem/1436/E) | MEX of all subarrays, then take the MEX again                          |
| 23   | [AtCoder ABC330E: Mex and Update](https://atcoder.jp/contests/abc330/tasks/abc330_e)           | Support element modification or querying full array MEX                |
| 24   | [Luogu P11837: Making Mexes B](https://www.luogu.com.cn/problem/P11837)                        | Minimum edits to ensure $\mathrm{mex}(a) = i$                        |
