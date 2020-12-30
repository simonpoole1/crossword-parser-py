
def my_func(label, array_arg):
    print("%s:" % label)
    for i in array_arg:
        print("%s" % i)

my_func("tuple", ("a", "b"))
my_func("list", ["a", "b"])

tuple_var = ("a", "b")
my_func("tuple var", tuple_var)
list_var = [ "a", "b"]
my_func("list var", list_var)

