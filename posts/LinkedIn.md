02.09.2026

𝗶𝗻𝗱𝗲𝘅𝗲𝗱-𝗽𝗮𝗿𝗾𝘂𝗲𝘁-𝗱𝗮𝘁𝗮𝘀𝗲𝘁

Today, I found a memory leak when using the module, so I had to fix it quickly.

𝗙𝗜𝗫𝗘𝗗:
• Fixed memory leak and I/O thrashing in PyArrow caused by sequential row-by-row fetching by implementing batched __getitems__ in __iter__.
• Fixed slow sequential fetching in train_test_split(stratify_by=...) by switching to the optimized __iter__ implementation.
https://pypi.org/project/indexed-parquet-dataset/
