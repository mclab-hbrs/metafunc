# metafunc: C++17 Functional Template Metaprogramming Library

metafunc is a functional template metaprogramming library, translating concepts found in functional languages to template metaprogramming. It is implemented in a single header file.

## Features

- Functions as First-Class Values

- Lazy Evaluation

- Algebraic Data Types

- Infinite Data Structures

- Type Classes

- Lambda Expressions

## Example Usage

```cpp
#include <metafunc.hpp>
using namespace metafunc::all;

// sort a list of ints
static_assert(std::is_same_v<
    eval<sort_<vals_int<3, 1, 2>>>,
    eval<vals_int<1, 2, 3>>
>);

// add pointers to a list of types
using add_ptr = func_from_templ<std::add_pointer_t>;
static_assert(std::is_same_v<
    eval<map_<add_ptr, make_list_<char, void, int>>>,
    eval<make_list_<char *, void *, int *>>
>);


// sort types by size and create tuple

// comparison function for sort
template<typename A, typename B>
using less_sz_ = val_bool<(sizeof(eval<A>) < sizeof(eval<B>))>;

// helper for turning a list into a std::tuple
template<typename Tuple, typename T>
struct tuple_append_ : tuple_append_<typename Tuple::type, T> {};
template<typename T, typename... Ts>
struct tuple_append_<std::tuple<Ts...>, T> {
    using type = std::tuple<Ts..., T>;
};

// sort list
using list_sorted = sort_by_<func_from_templ<less_sz_>, make_list_<int, short, long>>;
// convert to std::tuple
using tuple = fold_left_<func_from_templ<tuple_append_>, std::tuple<>, list_sorted>;
// eval and get result type
using result = eval<tuple>::type;
static_assert(std::is_same_v<result, std::tuple<long, int, short>>);


// count number of primes less than 50

// increment number
using increment = partial_<add, val_int<1>>;

// is A divisible by B?
template<typename ValA, typename ValB>
using divisible_ = val_bool<eval<ValA>::value % eval<ValB>::value == 0>;
using divisible = func_from_templ<divisible_>;

// get last element of list
using last = compose_<head, reverse>;

// is number less than 50?
using less_than_50 = lambda<less_<_1, val_int<50>>>;

// infinite list of positive integers
using integers = iterate_<increment, val_int<2>>;

// given a list of primes, choose
// the smallest integer not divisible by any prime
// as next prime
template<typename Primes>
using next_prime_ =
    head_<filter_<
        lambda<all_<map_<
            compose_<logic_not, partial_<divisible, _1>>,
            Primes>>>,
        integers>>;

// given a list of primes, append the next prime to the list
using add_next_prime = lambda2<p, append_<next_prime_<p>, p>>;

// infinite list of primes:
// start with prime 2, iterate add_next_prime, take the last element from each list
using primes = map_<last, iterate_<add_next_prime, vals_int<2>>>;

// count number of primes less than 50
using number = count_<take_while_<less_than_50, primes>>;
static_assert(eval<number>::value == 15);
```

For more examples on how to use this library, take a look at the tests inside the `metafunc::test` namespace.

## Concepts

### Types and Values

In template metaprogramming, everything operates on types, so C++-types become meta-values. To store C++-values as meta-values, they are wrapped in a type like `std::integral_constant`:

```cpp
template<typename T, T V>
struct val {
    static constexpr T value = V;
};
```

Additional aliases for common types are provided:

```cpp
template<std::size_t Val>
using val_size = val<std::size_t, Val>;

template<int Val>
using val_int = val<int, Val>;

template<bool Val>
using val_bool = val<bool, Val>;
```

### Functions as First-Class Values

Functions in metaprogramming are usually implemented as type templates:

```cpp
template<typename A, typename B>
struct add : val<decltype(A::value + B::value), A::value + B::value> {};
```

But in metafunc, these templates are wrapped in another type, so that they become (meta-)values themselves:

```cpp
struct add {
    template<typename A, typename B>
    struct apply : val<decltype(A::value + B::value), A::value + B::value> {};
};
```

This enables higher-order functions, i.e. functions with other functions as arguments or return value. For example, the `constant` function takes an argument and returns a function which always returns this argument:

```cpp
struct constant {
    template<typename Val>
    struct apply {
        template<typename>
        struct apply : Val {};
    };
};
```

For every function, a convenience alias to its `apply` member is defined, with a trailing underscore in its name:

```cpp
template<typename A, typename B>
using add_ = add::apply<A, B>;

template<typename Val>
using constant_ = constant::apply<Val>;
```

### Lazy Evaluation

All values in metafunc are lazily evaluated by default. These "lazy" values represent a computation not yet performed, and can be evaluated on demand using the `eval` template.

```cpp
static_assert(eval<mul_<val_int<2>, val_int<3>>>::value == 6);
```

### Algebraic Data Types

Algebraic data types are only represented by their constructors. Constructors in turn are implemented as functions. For example, the list data type has two constructors, `nil_<>` and `cons_<Head, Tail>`.

Partial template specializations of function's `apply` template member can be used as a basic form of pattern matching. Using this in conjunction with lazy evaluation requires the `data` template, which partially evaluates a lazy value to determine its data type.

```cpp
struct is_empty {
    // List is a lazy value, convert it to either nil_<> or cons_<Head, Tail> using data<>
    template<typename List>
    struct apply : apply<data<List>> {};

    // nil_<> is empty
    template<>
    struct apply<nil_<>> : val_bool<true> {};

    // cons_<Head, Tail> is not empty
    template<typename Head, typename Tail>
    struct apply<cons_<Head, Tail>> : val_bool<false> {};
};

// convenience alias
template<typename List>
using is_empty_ = is_empty::apply<List>;

static_assert(eval<is_empty_<nil_<>>>::value == true);
static_assert(eval<is_empty_<cons_<int, nil_<>>>>::value == false);
```

### Infinite Data Structures

Using lazy evaluation, infinitely large data structures can be created:

```cpp
struct repeat {
    template<typename Value>
    struct apply
        : cons_<Value, repeat_<Value>> {};
};
```

`repeat_<int>` will evaluate to an infinite list whose elements are all `int`.

### Type Classes

Type classes provide an abstraction of data types by requiring certain operations to be implemented for these data types. In metafunc, operations on data types are implemented as (meta-)functions nested inside the data type. Where possible, non-nested functions are provided, which delegate to the nested function depending on their argument:

```cpp
struct map {
    template<typename Func, typename Functor>
    struct apply : Functor::map::apply<Func, Functor> {};
};
```

### Lambda Expressions

Lambda expressions provide a way to define unnamed functions with a convenient syntax. metafunc provides two kinds of lambda expressions, `lambda` with implicit arguments and `lambda2` with explicitly named arguments. Both of them take a single expression and return a function, whose return value is the expression with all argument placeholders replaced with the actual arguments of the function call.

For `lambda`, the placeholders are `_1` to `_9` in the `util::args` namespace. `lambda`s can be nested, but an inner `lambda`'s body can only reference its own arguments and not the arguments of outer `lambda`s. This limitation does not apply to `lambda2`. The placeholders for `lambda2` are `a` to `z` in the `util::vars` namespace, and any placeholder used in the in the body of a `lambda2` has to be passed as an additional template argument. For nested `lambda2`s, the usual lexical scoping rules apply.

```cpp
// identity(_1) = _1
using identity = lambda<_1>;

// constant(a) = (b -> a)
using constant = lambda2<a, lambda2<b, a>>;

// logic_or(a, b) = if a then true else b
using logic_or = lambda2<a, b, cond_<a, val_bool<true>, b>>;
```

## Member Listing

- utilities (namespace `util`)
  - `eval`: evaluate value
  - `uneval`: wrap value so it can be evaluated
  - `data`: convert to instance of data type
  - `func_from_templ`: convert template to meta-function
  - `val`, `val_size`, `val_int`, `val_bool`: wrappers for values
  - `vals`, `vals_size`, `vals_int`, `vals_bool`: lists of values
  - `lambda`, `lambda2`: make function from expression with argument placeholders
- data types (namespace `data`)
  - `nil`, `cons`: list type
  - `nothing`, `just`: maybe type
  - `pair`: pair type
- functions (namespace `func`)
  - `apply`: apply a function to some arguments
  - operations on data types
    - `map`: map a function over a functor
    - `join`: flatten a monad of monads
    - `bind`: combine monad with a monadic function
  - unpack data types
    - `head`: get head of list
    - `tail`: get tail of list
    - `first`: get first element of pair
    - `second`: get second element of pair
  - basic functions
    - `cond`: conditional
    - `make_list`: make list from arguments
    - `id`: identity function
    - `constant`: create constant function
    - `equal`: test if two values are equal
  - higher-order functions
    - `fix_arg_count`: force number of arguments of a function
    - `partial`: bind arguments to a function
    - `y_comb`: higher-order function to enable recursive lambdas, inspired by the y combinator
    - `arg_count`: count number of arguments of function
    - `curry_n`: enable currying for a function
    - `curry`: enable currying for a function
    - `compose`: compose functions
  - arithmetic functions
    - `add`: add any number of values
    - `sub`: subtract two values
    - `mul`: multiply any number of values
    - `div`: divide two values
    - `greater`: test if one value is greater than another
    - `less`: test if one value is less than another
    - `greater_equal`: test if one value is greater than or equal to another
    - `less_equal`: test if one value is less than or equal to another
  - logic functions
    - `logic_and`: logic and, short-circuiting
    - `logic_or`: logic or, short-circuiting
    - `logic_not`: logic not
  - operations on lists
    - `fold_left`: fold function left over list
    - `fold_right`: fold function right over list
    - `mox_by`: get maximum of list given a comparison function
    - `iterate`: generate list by iterating function
    - `repeat`: generate list by repeating value
    - `concat`: concatenate two lists
    - `filter`: filter list by predicate
    - `at`: get element of list
    - `append`: append element to list
    - `prepend`: prepend element in front of list
    - `reverse`: reverse list
    - `take`: take elements from beginning of list
    - `drop`: drop elements from beginning of list
    - `take_while`: take elements from beginning of list
    - `drop_while`: drop elements from beginning of list
    - `index_of`: get index of element in list
    - `zip_with`: zip two lists and map a function over the result
    - `max`: get maximum of list
    - `min`: get minimum of list
    - `all`: and all values of a list
    - `any`: or all values of a list
    - `sum`: add all values of a list
    - `count`: get length of list
    - `count_if`: get number of values satisfying predicate
    - `contains`: test if list contains value
    - `partition`: partition list according to predicate
    - `sort_by`: sort list by comparison function
    - `sort`: sort list ascending

Additionally, the namespace `all` includes all of `util`, `data` and `func`.
