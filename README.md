# Vision Language Models Are Blind

<div align="center">    
    <p style="font-size: 45px;"> by 
        <a href="https://pooyanrg.me">Pooyan Rahmanzadehgervi</a><sup>1,*</sup>, Logan Bolton<sup>1,*</sup>,
        <a href="https://taesiri.ai">Mohammad Reza Taesiri</a><sup>2</sup>, 
        <a href="https://anhnguyen.me/research/">Anh Totti Nguyen</a><sup>1</sup>
    </p>
    <p>
        <sup>*</sup>Equal contribution<br>
        <sup>1</sup>Auburn University, <sup>2</sup>University of Alberta
    </p>
</div>

This repository contains the code and data for the paper `Vision Language Models Are Blind`.

## Abstract

*Large language models with vision capabilities (VLMs), e.g., GPT-4o and Gemini-1.5 Pro, are powering countless image-text processing applications and scoring high on existing vision-understanding benchmarks. Yet, we find that VLMs fail on 8 visual tasks that are absurdly easy for humans, such as identifying (a) whether two circles overlap; (b) whether two lines intersect; (c) which letter is being circled in a word; and (d) counting the number of circles in an Olympic-like logo. The shockingly poor performance of four state-of-the-art VLMs suggests their vision is, at best, like that of a person with myopia seeing fine details as blurry, and at worst, like an intelligent person who is blind making educated guesses.*

## Tasks in the BlindTest benchmark

1. [Task 1: Counting Line Intersection](./src/LineIntersection/)
1. [Task 2: Two Circles](./src/TouchingCircle/)
1. [Task 3: Circled Letter](./src/CircledWord/)
1. [Task 4: Counting Circles](./src/CountingCircles/)
1. [Task 5: Counting Nested Squares](./src/NestedSquares/)
1. [Task 6: Counting Rows and Columns](./src/CountingRowsAndColumns/)
1. [Task 7: Following color-coded paths](./src/SubwayMap/)



## Benchmark Results


### Task 1 - Counting Line Intersection

| Thickness | GPT-4o | Gemini-1.5 Pro | Sonnet 3.0 | Sonnet 3.5 |
|---------------|--------|----------------|------------|------------|
| 2             | 45.00  | 70.00          | 64.00      | 80.00      |
| 3             | 47.00  | 68.00          | 66.00      | 79.00      |
| 4             | 54.00  | 71.00          | 62.00      | 73.00      |
| **Average**   | 48.67  | 69.67          | 64.00      | **77.33**  |



![vision-llms-are-blind](./Figures/2Dlines-aibox.png)


### Task 2 - Two Circles



|                | GPT-4o | Gemini-1.5 Pro | Sonnet 3.0 | Sonnet 3.5 |
|----------------|--------|----------------|------------|------------|
| Overlapping    | 71.27  | **93.30**      | 88.09      | 88.83      |
| Touching       | 74.10  | 92.26          | 80.95      | **94.49**  |
| Average        | 72.69  | **92.78**      | 84.52      | 91.66      |


![vision-llms-are-blind](./Figures/2Touching-aibox.png)


### Task 3 -  Circled Letter

| Word                           | GPT-4o | Gemini-1.5 Pro | Sonnet 3.0 | Sonnet 3.5 |
|--------------------------------|--------|----------------|------------|------------|
| Acknowledgement                | 69.03  | 97.50          | 82.64      | 91.11      |
| Subdermatoglyphic              | 63.60  | 91.05          | 71.45      | 94.49      |
| tHyUiKaRbNqWeOpXcZvM           | 77.92  | 89.90          | 65.94      | 82.08      |
| **Average**                    | 70.18  | 92.81          | 73.34      | 89.22      |


![vision-llms-are-blind](./Figures/Redoval-aibox.png)


### Task 4 & 5 - Counting Circles and Nested Squares

|          | GPT-4o | Gemini-1.5 Pro | Sonnet 3.0 | Sonnet 3.5     |
|----------|--------|----------------|------------|----------------|
| Squares  | 48.33  | 80.00          | 55.00      | **87.50**      |
| Circles  | 42.50  | 20.83          | 31.66      | **44.16**      |
| Pentagons| 19.16  | 9.16           | 11.66      | **75.83**      |


![vision-llms-are-blind](./Figures/Nested-aibox.png)

![vision-llms-are-blind](./Figures/Olympic-aibox.png)


### Task 6 - Counting Rows and Columns


| Grid type | GPT-4o       | Gemini-1.5 Pro | Sonnet 3.0    | Sonnet 3.5      |
|-----------|--------------|----------------|---------------|-----------------|
| Blank     | 26.13        | 25.75          | 25.00         | 59.84           |
| Text      | **53.03**    | **45.83**      | **47.34**     | **88.68**       |
| Average   | 39.58        | 35.79          | 36.17         | 74.26           |


![vision-llms-are-blind](./Figures/Grid-aibox.png)


### Task 7 -Following color-coded paths

| Paths  | GPT-4o           | Gemini-1.5 Pro | Sonnet 3.0     | Sonnet 3.5      |
|--------|------------------|----------------|----------------|-----------------|
| 1      | 67.50            | 85.41          | 23.75          | **95.00**       |
| 2      | 44.37            | 28.75          | 37.18          | **56.25**       |
| 3      | **36.71**        | 25.78          | 15.42          | 25.39           |
| Average| 45.89            | 40.01          | 23.78          | **50.18**       |


![vision-llms-are-blind](./Figures/Subway-aibox.png)
