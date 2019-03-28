def print_VARS(v, precision = 0, predicates = []):
    (sem, struct) = v
    format_str = "{:>" + str(precision + 3) + "." + str(precision) + "f}"
        
    print("\n")
    if len(predicates):
        predicates_str = map(lambda(x): str(x) if x <> None else "", predicates)

    for wm_i in range(struct.shape[1]):
        print(
                " ".join(map(lambda(x): format_str.format(x), sem[wm_i]))
                +
                "    "
                +
                "    ".join(            
                    [" ".join(map(lambda(x): format_str.format(x), x)) for x in [struct_ai[wm_i] for struct_ai in struct]]
                )        
                +
                ("    " + predicates_str[wm_i] if len(predicates) else "") 
            )            
    print("\n")