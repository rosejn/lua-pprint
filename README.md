# Pretty Printing for Torch and Lua.

pprint(foo) will print a human readable printout of Lua tables.

* keys are printed alphabetically
* nested tables are indented
* loops are detected
* metatable info is printed
* torch tensors are printed if less than 20 elements, otherwise just the dimensions.
