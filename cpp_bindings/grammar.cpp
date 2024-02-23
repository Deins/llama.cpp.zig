#include <iostream>
#include <grammar-parser.h>

extern "C" llama_grammar * parse_grammar_from_text(const char * str) {
    grammar_parser::parse_state parsed_state = grammar_parser::parse(str);
    std::vector<const llama_grammar_element *> grammar_rules(parsed_state.c_rules());

    return llama_grammar_init(grammar_rules.data(), grammar_rules.size(), parsed_state.symbol_ids.at("root"));
}