# PPCompliance
This repository contains data for ASE 2022 submission: Are they Toeing the Line? Auditing Privacy Compliance among Browser Extensions

## Data Set
The full list for all the extensions we crawled from the Chrome Web Store is located in:
```
./chrome_60k_fulllist.json
```
Meanwhile, we collected all possible meta data during the crawling, including the id, name, author, subcategory, downlaods, rating, introduction, last update time, privacy policy declared, and outside privacy policy link.

The source code for each extension and corresponding privacy HTML file is located in Dropbox shared folder:
https://www.dropbox.com/sh/vq22x69pn5etl22/AAABcN9RYfcZSjPnlcdyMvRsa?dl=0
### Source Code
```
[shared_folder]/source_code/[ext_id].crx
```
There are 64,114, extensions with source code, 66G, in total.
### Privacy Policy Files
The raw HTML file for each linked privacy policy pages is located in the directory:
```
[shared_folder]/policy_pages/[ext_id].crx
```
There are 20,761 HTML files, 846M, in total.
