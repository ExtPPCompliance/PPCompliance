# PPCompliance
This repository contains data for ASE 2022 submission: Are they Toeing the Line? Auditing Privacy Compliance among Browser Extensions

## Data Set
The full list for all the extensions we crawled from the Chrome Web Store is located in:
```
./chrome_60k_fulllist.json
```
Meanwhile, we collected all possible meta data during the crawling, including the id, name, author, subcategory, downlaods, rating, introduction, last update time, privacy policy declared, and outside privacy policy link.

The source code for each extensions is located in the directory:
```
./source_code/[ext_id].crx
```
There are 64,114, 66G extensions with source code in total.
The raw HTML file for each linked privacy policy pages is located in the directory:
```
./policy_pages/[ext_id].crx
```
There are 20,761, 846M HTML files in total.
