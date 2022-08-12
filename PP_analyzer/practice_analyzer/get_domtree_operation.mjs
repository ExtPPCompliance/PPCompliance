import esprima from 'esprima'
import fs from 'fs';
import estraverse from 'estraverse';

function begin() {
    var walk = function (dir) {
        var results = [];
        var list = fs.readdirSync(dir);
        list.forEach(function (file) {
            if (file != '.DS_Store') {
                file = dir + '/' + file;
                var stat = fs.statSync(file);
                if (stat && stat.isDirectory()) {
                    /* Recurse into a subdirectory */
                    results = results.concat(walk(file));
                } else {
                    /* Is a file */
                    results.push(file);
                }
            }
        });
        return results;
    }

    function traverse_file(dir,) {
        var js_file = walk(dir);

        js_file.forEach(function (p) {
            if (p.slice(-3) == ".js") {
                // console.log(p);
                var output = p.slice(0, -3) + '_dom_tree.json'
                handle_ast_file(p, output, traverse);
            }
        });

        return;
    }
    function handle_ast_file(input_file, output_file, traverse) {
        var data = fs.readFileSync(input_file, { encoding: 'utf-8' });

        try {
            var node = esprima.parseScript(data);

            var called_api = []

            traverse(node, called_api, handle_ast_node);
            console.log(called_api);
            if (fs.existsSync(output_file)) {
                fs.unlinkSync(output_file);
            }
            fs.writeFileSync(output_file, JSON.stringify(called_api));
            success = success + 1;
            // console.log(input_file)
            console.log(success);
        } catch (e) {
            // console.error('skip a file' + input_file);
            console.log(e);
            failed = failed + 1;
        }

    }
    function traverse(node, called_api, handle_ast_node) {
        var parentChain = [];
        estraverse.traverse(node, {
            enter: (node, parent) => {
                parentChain.push(parent);
                handle_ast_node(node, called_api, parentChain);
            },
            leave: (node) => {
                parentChain.pop();
            }
        }
        );

    }

    function handle_ast_node(node, called_api, parentChain) {

        // reached_node.push(node);
        // console.log("handle one node")
        try {
            var recallChain = [];
            var partFunc = node.name + ".";
            if (node.name == "document") {
                while (parentChain.length != 0) {
                    var parent = parentChain.pop();
                    recallChain.push(parent);
                    if (parent.type == "CallExpression") {
                        // reach the end
                        console.log("length of dom tree api " + recallChain.length);
                        // console.log("name of chrome API"+JSON.stringify(parent));
                        partFunc = partFunc.slice(0, -1);
                        console.log("name of API ", partFunc);
                        called_api.push({ "name": partFunc, "arguments": parent.arguments })
                        break;
                    } else if (parent.type == "MemberExpression") {
                        partFunc = partFunc + parent.property.name + '.';

                    } else {
                        // unqualified
                        break;
                    }
                    // console.log("get inside " + JSON.stringify(parent));
                }
                while (recallChain.length != 0) {
                    parentChain.push(recallChain.pop());
                }
            } else {
                // console.log(parentChain.length);
            }


        } catch (e) {
            console.log(e);
        }

    }
    function print_info() {
        console.log('failed' + failed);
        console.log('success' + success);
        return;
    }
    var success = 0;
    var failed = 0;

    // print_info();
    traverse_file('../raw_data/process', handle_ast_file);
    print_info();
}
begin();

// var input_file="sample_ext/process/bhghoamapcdpbohphigoooaddinpkbai/dist/import.js";
// var output_file="";
// handle_ast_file(input_file,output_file)