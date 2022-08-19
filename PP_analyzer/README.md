## Instructions

The PP analyzer is consisted with `privacy analyzer` and `practice analyzer`. 

### Privacy Analyzer
For `privacy_analyzer`, it gives the result predicted by our privacy policy analysis model.
Get the prediction result, run the following code:
```
python3 ./privacy_analyzer/large_scale_infer.py
```
The input file is the original privacy policy HTML pages. Put them all in the directory which is defined in the code. Then run the commond above.
You will get the result at:
```
./privacy_analyzer/privacy_conclude_result_last.csv
```
The result will be transferred to large scale analysis to check the compliance issues among the whole store.
If you want to evaluate a single extension or get the analysis result immediately, the code is easy extendable for your further implementation.

### Practice Analyzer
For `practice_analyzer`, it gives the actual privacy practice for all the extensions.
The input file is the original extension compresed source code package with `crx` suffix.
The whole analysis process is shown below:
#### Unzip the CRX file
Put all the `crx` files are located in a directory named `zip`, run the command below:
```
sh ./practice_analyzer/unzip.sh [$your_directory_of_zip_files]
```
You will get the unzip files in the same directory names `unzip`
#### Get Chrome API Call and DOM Operation API Call
Change the path to your unzip folder in line 128 of `./practice_analyzer/get_chrome_api.mjs`, then run the code with `node` command.

The same operation to `./practice_analyzer/get_domtree_operation.mjs`, and run it with `node`.

After that, your `unzip` folder will contain two more json file for each JavaScript file ends with `_chrome_api.json` and `_dom_tree.json`. Further analysis is based on these files.

To extract the callee details, comment line 151 or line 152 of `./practice_analyzer/get_callee.py`. Run it respectively to get the API call results for each extension.

The result files are in the format as `[ext_id]_privacy_api.json` and `[ext_id]_dom_operation.json`.

#### Get Target Tags from Dynamic Pages
Change the path to your dynamic pages folder in line 46 of `./practice_analyzer/get_tag.py`, then run it with `python3`.

The result files are in the format as `[ext_id]_userinput_tags.json`.

#### Get Privacy Compliance Conclusion
Change the path to your unzip folder or dynamic pages folder in line 86 of `./practice_analyzer/conclude_all_results.py`, then run it with  python3`.

Uncomment the code to get both Chrome API result and dynamic input tag result.

The result files are `./practice_analyzer/chrome_api_conclude.csv` and `./practice_analyzer/dynamic_input_tag_conclude.csv`.

Based on the result files, you can do whatever analysis you want. 

### Conclusion
We've implement the whole Chrome Web Store analysis. Our code is easy to be extended and reused.
You can easily analyze a single extensions or achieve a large scale analysis based on our code.
If you have any question, feel free to contact us.

That all for our analysis tools, enjoy your research!
