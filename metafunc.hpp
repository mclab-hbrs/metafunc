#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace metafunc {

    // utility types and templates which are not meta-functions
    namespace util {

        // eval: evaluate value
        namespace detail {
            template<typename T, typename = void>
            struct eval_impl {
                using result = T;
            };
            template<typename T>
            struct eval_impl<T, std::void_t<typename T::eval>> {
                using result = typename T::eval::result;
            };
        } // namespace detail
        template<typename T>
        using eval = typename detail::eval_impl<T>::result;

        // uneval: wrap value so it can be evaluated
        namespace detail {
            template<typename T, typename = void>
            struct uneval_impl {
                struct eval {
                    using result = T;
                };
            };
            template<typename T>
            struct uneval_impl<T, std::void_t<typename T::eval>> : T {};
        } // namespace detail
        template<typename T>
        using uneval = detail::uneval_impl<T>;

        // data: convert to instance of data type
        template<typename T>
        using data = typename T::data::result;

        // func_from_templ: convert template to meta-function
        template<template<typename...> typename Templ,
                std::size_t ArgCount = std::numeric_limits<std::size_t>::max()>
        struct func_from_templ {
            template<typename, typename... Args>
            struct impl : uneval<Templ<eval<Args>...>> {};
            template<typename... Args>
            using apply = impl<std::enable_if_t<sizeof...(Args) == ArgCount
                                       || ArgCount == std::numeric_limits<std::size_t>::max()>,
                    Args...>;
        };

        // wrappers for values
        template<typename T, T Val>
        using val = std::integral_constant<T, Val>;
        template<std::size_t Val>
        using val_size = val<std::size_t, Val>;
        template<int Val>
        using val_int = val<int, Val>;
        template<bool Val>
        using val_bool = val<bool, Val>;

        // args for lambda
        template<std::size_t Idx>
        struct arg {};
        namespace args {
            using _1 = arg<0>;
            using _2 = arg<1>;
            using _3 = arg<2>;
            using _4 = arg<3>;
            using _5 = arg<4>;
            using _6 = arg<5>;
            using _7 = arg<6>;
            using _8 = arg<7>;
            using _9 = arg<8>;
        } // namespace args

        // vars for lambda2
        template<std::size_t Id>
        struct var {};
        namespace vars {
            using a = var<0>;
            using b = var<1>;
            using c = var<2>;
            using d = var<3>;
            using e = var<4>;
            using f = var<5>;
            using g = var<6>;
            using h = var<7>;
            using i = var<8>;
            using j = var<9>;
            using k = var<10>;
            using l = var<11>;
            using m = var<12>;
            using n = var<13>;
            using o = var<14>;
            using p = var<15>;
            using q = var<16>;
            using r = var<17>;
            using s = var<18>;
            using t = var<19>;
            using u = var<20>;
            using v = var<21>;
            using w = var<22>;
            using x = var<23>;
            using y = var<24>;
            using z = var<25>;
        } // namespace vars

        // ---------------- utils for packs ----------------

        namespace detail {
            // pack: used to hold a template parameter pack
            template<typename...>
            struct pack {};

            // at: get element of a pack
            template<std::size_t Idx, typename Pack>
            struct at_impl;
            template<std::size_t Idx, typename... Ts>
            struct at_impl<Idx, pack<Ts...>> {
                using type = std::tuple_element_t<Idx, std::tuple<Ts...>>;
            };
            template<std::size_t Idx, typename Pack>
            using at = typename at_impl<Idx, Pack>::type;

            // last: get last element of a pack
            template<typename Pack>
            struct last_impl;
            template<typename... Ts>
            struct last_impl<pack<Ts...>> {
                using type = at<sizeof...(Ts) - 1, pack<Ts...>>;
            };
            template<typename Pack>
            using last = typename last_impl<Pack>::type;

            // contains: check if a pack contains an element
            template<typename Elem, typename Pack>
            struct contains_impl;
            template<typename Elem>
            struct contains_impl<Elem, pack<>> : std::false_type {};
            template<typename Elem, typename T, typename... Ts>
            struct contains_impl<Elem, pack<T, Ts...>> : std::conditional_t<std::is_same_v<Elem, T>,
                                                                 std::true_type,
                                                                 contains_impl<Elem, pack<Ts...>>> {
            };
            template<typename Elem, typename Pack>
            inline constexpr bool contains = contains_impl<Elem, Pack>::value;

            // without_last: transform pack, using all elements but the last
            template<template<typename...> typename Templ,
                    typename Pack,
                    typename ExtraArgsFront,
                    typename ExtraArgsBack,
                    typename Is>
            struct without_last_impl;
            template<template<typename...> typename Templ,
                    typename... Ts,
                    typename... ExtraArgsFront,
                    typename... ExtraArgsBack,
                    std::size_t... Is>
            struct without_last_impl<Templ,
                    pack<Ts...>,
                    pack<ExtraArgsFront...>,
                    pack<ExtraArgsBack...>,
                    std::index_sequence<Is...>> {
                using type = Templ<ExtraArgsFront...,
                        std::tuple_element_t<Is, std::tuple<Ts...>>...,
                        ExtraArgsBack...>;
            };
            template<template<typename...> typename Templ,
                    typename Pack,
                    typename ExtraArgsFront,
                    typename ExtraArgsBack>
            struct without_last_impl_proxy;
            template<template<typename...> typename Templ,
                    typename... Ts,
                    typename ExtraArgsFront,
                    typename ExtraArgsBack>
            struct without_last_impl_proxy<Templ, pack<Ts...>, ExtraArgsFront, ExtraArgsBack>
                    : without_last_impl<Templ,
                              pack<Ts...>,
                              ExtraArgsFront,
                              ExtraArgsBack,
                              std::make_index_sequence<sizeof...(Ts) - 1>> {};
            template<template<typename...> typename Templ,
                    typename Pack,
                    typename ExtraArgsFront = pack<>,
                    typename ExtraArgsBack = pack<>>
            using without_last =
                    typename without_last_impl_proxy<Templ, Pack, ExtraArgsFront, ExtraArgsBack>::
                            type;
        } // namespace detail

        // ---------------- lambda ----------------

        // lambda: make function from expression with argument placeholders
        template<typename Expr>
        struct lambda;

        namespace detail {
            // substitute all occurrences of arg<Idx> in Expr with Args
            template<typename Expr, typename... Args>
            struct lambda_subst {
                using result = Expr;
            };
            // replace arg<Idx> when found
            template<std::size_t Idx, typename... Args>
            struct lambda_subst<arg<Idx>, Args...> {
                using result = eval<at<Idx, pack<Args...>>>;
            };
            // recursively replace in subexpressions
            template<template<typename...> typename Expr, typename... ExprArgs, typename... Args>
            struct lambda_subst<Expr<ExprArgs...>, Args...> {
                using result = Expr<typename lambda_subst<ExprArgs, Args...>::result...>;
            };
            // special case for nested lambdas
            template<typename Nested, typename... Args>
            struct lambda_subst<lambda<Nested>, Args...> {
                using result = lambda<Nested>;
            };
        } // namespace detail

        template<typename Expr>
        struct lambda {
            template<typename... Args>
            struct apply : uneval<typename detail::lambda_subst<Expr, Args...>::result> {};
        };

        // ---------------- lambda2 ----------------

        // lambda2: make function from expression with variable placeholders
        template<typename... Pack>
        struct lambda2;

        namespace detail {
            // substitute all occurrences of Var in Expr with Arg
            template<typename Expr, typename Var, typename Arg>
            struct lambda2_subst {
                using result = Expr;
            };
            // replace Var when found
            template<typename Var, typename Arg>
            struct lambda2_subst<Var, Var, Arg> {
                using result = Arg;
            };
            // recursively replace in subexpressions
            template<template<typename...> typename Expr,
                    typename... ExprArgs,
                    typename Var,
                    typename Arg>
            struct lambda2_subst<Expr<ExprArgs...>, Var, Arg> {
                using result = Expr<typename lambda2_subst<ExprArgs, Var, Arg>::result...>;
            };
            // special case for nested lambdas
            template<typename... LambdaArgs, typename Var, typename Arg>
            struct lambda2_subst<lambda2<LambdaArgs...>, Var, Arg> {
                using result = std::conditional_t<
                        contains<Var, without_last<pack, pack<LambdaArgs...>>>,
                        lambda2<LambdaArgs...>,
                        without_last<lambda2,
                                pack<LambdaArgs...>,
                                pack<>,
                                pack<typename lambda2_subst<last<pack<LambdaArgs...>>, Var, Arg>::
                                                result>>>;
            };

            template<typename Expr, typename PackVars, typename PackArgs>
            struct lambda2_impl;
            template<typename Expr>
            struct lambda2_impl<Expr, pack<>, pack<>> : uneval<Expr> {};
            template<typename Expr, typename Var, typename... Vars, typename Arg, typename... Args>
            struct lambda2_impl<Expr, pack<Var, Vars...>, pack<Arg, Args...>>
                    : lambda2_impl<typename lambda2_subst<Expr, Var, Arg>::result,
                              pack<Vars...>,
                              pack<Args...>> {};

        } // namespace detail

        template<typename... Pack>
        struct lambda2 {
            template<typename, typename... Args>
            struct impl : detail::lambda2_impl<detail::last<detail::pack<Pack...>>,
                                  detail::without_last<detail::pack, detail::pack<Pack...>>,
                                  detail::pack<Args...>> {};
            template<typename... Args>
            using apply = impl<std::enable_if_t<sizeof...(Args) + 1 == sizeof...(Pack)>, Args...>;
        };

    } // namespace util

    // data types and type classes
    namespace data {
        // operations on data types are here only declared and implemented later

        namespace detail {
            // ctor: superclass of all data type constructors, implementing data and eval
            template<typename T>
            struct ctor;
            template<template<typename...> typename T, typename... Args>
            struct ctor<T<Args...>> {
                // wrap aliases in type so we can access any one without instantiating the others

                struct data {
                    using result = T<Args...>;
                };

                struct eval {
                    using result = T<util::eval<Args>...>;
                };
            };
        } // namespace detail

        // ---------------- type classes ----------------

        template<typename Self>
        struct functor {
            struct map {
                template<typename Func, typename Functor>
                struct apply;
            };
        };

        template<typename Self>
        struct monad : functor<Self> {
            struct unit {
                template<typename Val>
                struct apply;
            };

            struct join {
                template<typename MonadMonad>
                struct apply;
            };

            struct bind {
                template<typename Monad, typename Func>
                struct apply;
            };
        };

        // ---------------- list ----------------

        struct list : monad<list> {
            struct map {
                template<typename Func, typename Functor>
                struct apply;
            };

            struct unit {
                template<typename Val>
                struct apply;
            };

            struct join {
                template<typename MonadMonad>
                struct apply;
            };
        };

        struct nil {
            template<typename = void>
            struct impl : list, detail::ctor<impl<>> {};
            template<typename... Empty>
            using apply = impl<std::enable_if_t<sizeof...(Empty) == 0>>;
        };
        template<typename... Empty>
        using nil_ = nil::apply<Empty...>;

        struct cons {
            template<typename Head, typename Tail>
            struct apply : list, detail::ctor<apply<Head, Tail>> {};
        };
        template<typename Head, typename Tail>
        using cons_ = cons::apply<Head, Tail>;

        // ---------------- maybe ----------------

        struct maybe : monad<maybe> {
            struct map {
                template<typename Func, typename Functor>
                struct apply;
            };

            struct unit {
                template<typename Val>
                struct apply;
            };

            struct join {
                template<typename MonadMonad>
                struct apply;
            };
        };

        struct nothing {
            template<typename = void>
            struct impl : maybe, detail::ctor<impl<>> {};
            template<typename... Empty>
            using apply = impl<std::enable_if_t<sizeof...(Empty) == 0>>;
        };
        template<typename... Empty>
        using nothing_ = nothing::apply<Empty...>;

        struct just {
            template<typename Val>
            struct apply : maybe, detail::ctor<apply<Val>> {};
        };
        template<typename Val>
        using just_ = just::apply<Val>;

        // ---------------- pair ----------------

        struct pair_type : functor<pair_type> {
            struct map {
                template<typename Func, typename Functor>
                struct apply;
            };
        };

        struct pair {
            template<typename First, typename Second>
            struct apply : pair_type, detail::ctor<apply<First, Second>> {};
        };
        template<typename First, typename Second>
        using pair_ = pair::apply<First, Second>;

    } // namespace data

    // meta-functions
    namespace func {

        // apply: apply a function to some arguments
        namespace detail {
            struct apply_impl {
                template<typename Func, typename... Args>
                struct apply : util::uneval<typename util::eval<Func>::template apply<Args...>> {};
            };
        } // namespace detail
        using apply = detail::apply_impl;
        template<typename Func, typename... Args>
        using apply_ = apply::apply<Func, Args...>;

        // ---------------- operations on data types ----------------

        // map: map a function over a functor
        struct map {
            template<typename Func, typename Functor>
            struct apply : apply_<typename Functor::map, Func, Functor> {};
        };
        template<typename Func, typename Functor>
        using map_ = apply_<map, Func, Functor>;

        // join: flatten a monad of monads
        struct join {
            template<typename MonadMonad>
            struct apply : apply_<typename MonadMonad::join, MonadMonad> {};
        };
        template<typename MonadMonad>
        using join_ = apply_<join, MonadMonad>;

        // bind: combine monad with a monadic function
        struct bind {
            template<typename Monad, typename Func>
            struct apply : apply_<typename Monad::bind, Monad, Func> {};
        };
        template<typename Monad, typename Func>
        using bind_ = apply_<bind, Monad, Func>;

        // ---------------- unpack data types ----------------

        // head: get head of list
        struct head {
            template<typename List>
            struct apply : apply<util::data<List>> {};
            template<typename Head, typename Tail>
            struct apply<data::cons_<Head, Tail>> : util::uneval<Head> {};
        };
        template<typename List>
        using head_ = apply_<head, List>;

        // tail: get tail of list
        struct tail {
            template<typename List>
            struct apply : apply<util::data<List>> {};
            template<typename Head, typename Tail>
            struct apply<data::cons_<Head, Tail>> : util::uneval<Tail> {};
        };
        template<typename List>
        using tail_ = apply_<tail, List>;

        // first: get first element of pair
        struct first {
            template<typename Pair>
            struct apply : apply<util::data<Pair>> {};
            template<typename First, typename Second>
            struct apply<data::pair_<First, Second>> : util::uneval<First> {};
        };
        template<typename Pair>
        using first_ = apply_<first, Pair>;

        // second: get second element of pair
        struct second {
            template<typename Pair>
            struct apply : apply<util::data<Pair>> {};
            template<typename First, typename Second>
            struct apply<data::pair_<First, Second>> : util::uneval<Second> {};
        };
        template<typename Pair>
        using second_ = apply_<second, Pair>;

        // ---------------- basic functions ----------------

        // cond: conditional
        struct cond {
            template<typename Condition, typename IfTrue, typename IfFalse>
            struct apply
                    : util::uneval<
                              std::conditional_t<util::eval<Condition>::value, IfTrue, IfFalse>> {};
        };
        template<typename Condition, typename IfTrue, typename IfFalse>
        using cond_ = apply_<cond, Condition, IfTrue, IfFalse>;

        // make_list: make list from arguments
        struct make_list {
            template<typename, typename... Vals>
            struct impl;
            template<typename Void>
            struct impl<Void> : data::nil_<> {};
            template<typename First, typename... Rest>
            struct impl<void, First, Rest...> : data::cons_<First, impl<void, Rest...>> {};

            template<typename... Vals>
            using apply = impl<void, Vals...>;
        };
        template<typename... Vals>
        using make_list_ = apply_<make_list, Vals...>;

        // id: identity function
        using id = util::lambda2<util::vars::a, util::vars::a>;
        template<typename Val>
        using id_ = apply_<id, Val>;

        // constant: create constant function
        using constant = util::lambda2<util::vars::a, util::lambda<util::vars::a>>;
        template<typename Val>
        using constant_ = apply_<constant, Val>;

        // equal: test if two values are equal
        using equal = util::func_from_templ<std::is_same, 2>;
        template<typename ValA, typename ValB>
        using equal_ = apply_<equal, ValA, ValB>;

        // ---------------- higher-order functions ----------------

        // fix_arg_count: force number of arguments of a function
        struct fix_arg_count {
            template<typename Func, typename Count>
            struct impl {
                template<typename, typename... Args>
                struct impl2 : apply_<Func, Args...> {};
                template<typename... Args>
                using apply = impl2<std::enable_if_t<sizeof...(Args) == util::eval<Count>::value>,
                        Args...>;
            };
            template<typename Func, typename Count>
            using apply = impl<Func, Count>;
        };
        template<typename Func, typename Count>
        using fix_arg_count_ = apply_<fix_arg_count, Func, Count>;

        // partial: bind arguments to a function
        struct partial {
            template<typename Func, typename... BoundArgs>
            struct impl {
                template<typename... Args>
                struct apply : apply_<Func, BoundArgs..., Args...> {};
            };
            template<typename Func, typename... BoundArgs>
            using apply = impl<Func, BoundArgs...>;
        };
        template<typename Func, typename... BoundArgs>
        using partial_ = apply_<partial, Func, BoundArgs...>;

        // y_comb: higher-order function to enable recursive lambdas, inspired by the y combinator
        namespace detail {
            template<typename X>
            using y2 = apply_<X, X>;
            template<typename F>
            using y1 =
                    util::lambda2<util::vars::x, partial_<F, apply_<util::vars::x, util::vars::x>>>;
        } // namespace detail
        using y_comb = util::lambda<detail::y2<detail::y1<util::arg<0>>>>;
        template<typename Func>
        using y_comb_ = apply_<y_comb, Func>;

        // arg_count: count number of arguments of function
        namespace detail {
            template<typename Func, typename Void, typename... Args>
            struct arg_count_impl : arg_count_impl<Func, Void, Args..., void> {};
            template<typename Func, typename... Args>
            struct arg_count_impl<Func,
                    std::void_t<typename Func::template apply<Args...>>,
                    Args...> : util::uneval<util::val_size<sizeof...(Args)>> {};
        } // namespace detail
        struct arg_count {
            template<typename Func>
            struct apply : detail::arg_count_impl<util::eval<Func>, void> {};
        };
        template<typename Func>
        using arg_count_ = apply_<arg_count, Func>;

        // curry_n: enable currying for a function
        namespace detail {
            template<typename Func, std::size_t Count, typename... CurArgs>
            struct curry_curry {
                template<typename NextArg>
                struct impl : std::conditional_t<sizeof...(CurArgs) + 1 == Count,
                                      apply_<Func, CurArgs..., NextArg>,
                                      curry_curry<Func, Count, CurArgs..., NextArg>> {};
                template<typename NextArg>
                using apply = impl<NextArg>;
            };

            template<typename Func, typename... Args>
            struct apply_curried;
            template<typename Func>
            struct apply_curried<Func> : util::uneval<Func> {};
            template<typename Func, typename Arg, typename... Args>
            struct apply_curried<Func, Arg, Args...> : apply_curried<apply_<Func, Arg>, Args...> {};
            template<typename Func>
            struct curry_uncurry {
                template<typename... Args>
                struct apply : apply_curried<Func, Args...> {};
            };

        } // namespace detail
        struct curry_n {
            template<typename Func, typename Count>
            struct impl
                    : detail::curry_uncurry<detail::curry_curry<Func, util::eval<Count>::value>> {};
            template<typename Func, typename Count>
            using apply = impl<Func, Count>;
        };
        template<typename Func, typename Count>
        using curry_n_ = apply_<curry_n, Func, Count>;

        // curry: enable currying for a function
        using curry =
                util::lambda2<util::vars::f, curry_n_<util::vars::f, arg_count_<util::vars::f>>>;
        template<typename Func>
        using curry_ = apply_<curry, Func>;

        // compose: compose functions
        struct compose {
            template<typename, typename... Funcs>
            struct impl;
            template<typename Void>
            struct impl<Void> : id {};
            template<typename Func, typename... Funcs>
            struct impl<void, Func, Funcs...> {
                template<typename Arg>
                struct apply : util::uneval<apply_<Func, apply_<impl<void, Funcs...>, Arg>>> {};
            };

            template<typename... Funcs>
            using apply = impl<void, Funcs...>;
        };
        template<typename... Funcs>
        using compose_ = apply_<compose, Funcs...>;

        // ---------------- arithmetic functions ----------------

        // add: add any number of values
        struct add {
            template<typename... Vals>
            struct apply : util::uneval<util::val<decltype((0 + ... + util::eval<Vals>::value)),
                                   (0 + ... + util::eval<Vals>::value)>> {};
        };
        template<typename... Vals>
        using add_ = apply_<add, Vals...>;

        // sub: subtract two values
        struct sub {
            template<typename ValA, typename ValB>
            struct apply
                    : util::uneval<
                              util::val<decltype(util::eval<ValA>::value - util::eval<ValB>::value),
                                      (util::eval<ValA>::value - util::eval<ValB>::value)>> {};
        };
        template<typename ValA, typename ValB>
        using sub_ = apply_<sub, ValA, ValB>;

        // mul: multiply any number of values
        struct mul {
            template<typename... Vals>
            struct apply : util::uneval<util::val<decltype((1 * ... * util::eval<Vals>::value)),
                                   (1 * ... * util::eval<Vals>::value)>> {};
        };
        template<typename... Vals>
        using mul_ = apply_<mul, Vals...>;

        // div: divide two values
        struct div {
            template<typename ValA, typename ValB>
            struct apply
                    : util::uneval<
                              util::val<decltype(util::eval<ValA>::value / util::eval<ValB>::value),
                                      (util::eval<ValA>::value / util::eval<ValB>::value)>> {};
        };
        template<typename ValA, typename ValB>
        using div_ = apply_<div, ValA, ValB>;

        // greater: test if one value is greater than another
        struct greater {
            template<typename ValA, typename ValB>
            struct apply
                    : util::uneval<
                              util::val_bool<(util::eval<ValA>::value > util::eval<ValB>::value)>> {
            };
        };
        template<typename ValA, typename ValB>
        using greater_ = apply_<greater, ValA, ValB>;

        // less: test if one value is less than another
        struct less {
            template<typename ValA, typename ValB>
            struct apply
                    : util::uneval<
                              util::val_bool<(util::eval<ValA>::value < util::eval<ValB>::value)>> {
            };
        };
        template<typename ValA, typename ValB>
        using less_ = apply_<less, ValA, ValB>;

        // greater_equal: test if one value is greater than or equal to another
        struct greater_equal {
            template<typename ValA, typename ValB>
            struct apply : util::uneval<util::val_bool<(
                                   util::eval<ValA>::value >= util::eval<ValB>::value)>> {};
        };
        template<typename ValA, typename ValB>
        using greater_equal_ = apply_<greater_equal, ValA, ValB>;

        // less_equal: test if one value is less than or equal to another
        struct less_equal {
            template<typename ValA, typename ValB>
            struct apply : util::uneval<util::val_bool<(
                                   util::eval<ValA>::value <= util::eval<ValB>::value)>> {};
        };
        template<typename ValA, typename ValB>
        using less_equal_ = apply_<less_equal, ValA, ValB>;

        // ---------------- logic functions ----------------

        // logic_and: logic and, short-circuiting
        using logic_and = util::lambda2<util::vars::a,
                util::vars::b,
                cond_<util::vars::a, util::vars::b, util::val_bool<false>>>;
        template<typename ValA, typename ValB>
        using logic_and_ = apply_<logic_and, ValA, ValB>;

        // logic_or: logic or, short-circuiting
        using logic_or = util::lambda2<util::vars::a,
                util::vars::b,
                cond_<util::vars::a, util::val_bool<true>, util::vars::b>>;
        template<typename ValA, typename ValB>
        using logic_or_ = apply_<logic_or, ValA, ValB>;

        // logic_not: logic not
        struct logic_not {
            template<typename Val>
            struct apply : util::uneval<util::val_bool<(!util::eval<Val>::value)>> {};
        };
        template<typename Val>
        using logic_not_ = apply_<logic_not, Val>;

        // ---------------- operations on lists ----------------

        // fold_left: fold function left over list
        struct fold_left {
            template<typename Func, typename Init, typename List>
            struct apply : apply<Func, Init, util::data<List>> {};
            template<typename Func, typename Init>
            struct apply<Func, Init, data::nil_<>> : util::uneval<Init> {};
            template<typename Func, typename Init, typename Head, typename Tail>
            struct apply<Func, Init, data::cons_<Head, Tail>>
                    : apply<Func, apply_<Func, Init, Head>, Tail> {};
        };
        template<typename Func, typename Init, typename List>
        using fold_left_ = apply_<fold_left, Func, Init, List>;

        // fold_right: fold function right over list
        struct fold_right {
            template<typename Func, typename Init, typename List>
            struct apply : apply<Func, Init, util::data<List>> {};
            template<typename Func, typename Init>
            struct apply<Func, Init, data::nil_<>> : util::uneval<Init> {};
            template<typename Func, typename Init, typename Head, typename Tail>
            struct apply<Func, Init, data::cons_<Head, Tail>>
                    : apply_<Func, Head, apply<Func, Init, Tail>> {};
        };
        template<typename Func, typename Init, typename List>
        using fold_right_ = apply_<fold_right, Func, Init, List>;

        // mox_by: get maximum of list given a comparison function
        namespace detail {
            template<typename Cmp>
            using choose_greater = util::lambda<
                    cond_<apply_<Cmp, util::arg<0>, util::arg<1>>, util::arg<0>, util::arg<1>>>;
        } // namespace detail
        template<typename Cmp, typename List>
        using max_by_ = fold_left_<detail::choose_greater<Cmp>, head_<List>, tail_<List>>;
        using max_by = util::func_from_templ<max_by_, 2>;

        // iterate: generate list by iterating function
        struct iterate {
            template<typename Func, typename Val>
            struct apply : data::cons_<Val, apply<Func, apply_<Func, Val>>> {};
        };
        template<typename Func, typename Val>
        using iterate_ = apply_<iterate, Func, Val>;

        // repeat: generate list by repeating value
        using repeat = fix_arg_count_<partial_<iterate, id>, util::val_size<1>>;
        template<typename Val>
        using repeat_ = apply_<repeat, Val>;

        // concat: concatenate two lists
        struct concat {
            template<typename ListA, typename ListB>
            struct apply : apply<util::data<ListA>, ListB> {};
            template<typename ListB>
            struct apply<data::nil_<>, ListB> : util::uneval<ListB> {};
            template<typename Head, typename Tail, typename ListB>
            struct apply<data::cons_<Head, Tail>, ListB> : data::cons_<Head, apply<Tail, ListB>> {};
        };
        template<typename ListA, typename ListB>
        using concat_ = apply_<concat, ListA, ListB>;

        // filter: filter list by predicate
        namespace detail {
            template<typename Pred>
            using filter_map_func = util::lambda<
                    cond_<apply_<Pred, util::arg<0>>, make_list_<util::arg<0>>, data::nil_<>>>;
        } // namespace detail
        template<typename Pred, typename List>
        using filter_ = join_<map_<detail::filter_map_func<Pred>, List>>;
        using filter = util::func_from_templ<filter_, 2>;

        // at: get element of list
        namespace detail {
            template<std::size_t Idx, typename List>
            struct at_impl;
            template<std::size_t Idx, typename Head, typename Tail>
            struct at_impl<Idx, data::cons_<Head, Tail>> : at_impl<Idx - 1, util::data<Tail>> {};
            template<typename Head, typename Tail>
            struct at_impl<0, data::cons_<Head, Tail>> : util::uneval<Head> {};
        } // namespace detail
        struct at {
            template<typename Idx, typename List>
            struct apply : detail::at_impl<util::eval<Idx>::value, util::data<List>> {};
        };
        template<typename Idx, typename List>
        using at_ = apply_<at, Idx, List>;

        // append: append element to list
        struct append {
            template<typename Val, typename List>
            struct apply : apply<Val, util::data<List>> {};
            template<typename Val, typename Head, typename Tail>
            struct apply<Val, data::cons_<Head, Tail>>
                    : data::cons_<Head, apply<Val, util::data<Tail>>> {};
            template<typename Val>
            struct apply<Val, data::nil_<>> : data::cons_<Val, data::nil_<>> {};
        };
        template<typename Val, typename List>
        using append_ = apply_<append, Val, List>;

        // prepend: prepend element in front of list
        using prepend = data::cons;
        template<typename Val, typename List>
        using prepend_ = apply_<prepend, Val, List>;

        // reverse: reverse list
        using reverse =
                fix_arg_count_<partial_<fold_right, append, data::nil_<>>, util::val_size<1>>;
        template<typename List>
        using reverse_ = apply_<reverse, List>;

        // take: take elements from beginning of list
        namespace detail {
            template<std::size_t Count, typename List>
            struct take_impl;
            template<std::size_t Count, typename Head, typename Tail>
            struct take_impl<Count, data::cons_<Head, Tail>>
                    : data::cons_<Head, take_impl<Count - 1, util::data<Tail>>> {};
            template<std::size_t Count>
            struct take_impl<Count, data::nil_<>> : data::nil_<> {};
            template<typename Head, typename Tail>
            struct take_impl<0, data::cons_<Head, Tail>> : data::nil_<> {};
        } // namespace detail
        struct take {
            template<typename Count, typename List>
            struct apply : detail::take_impl<util::eval<Count>::value, util::data<List>> {};
        };
        template<typename Count, typename List>
        using take_ = apply_<take, Count, List>;

        // drop: drop elements from beginning of list
        namespace detail {
            template<std::size_t Count, typename List>
            struct drop_impl;
            template<std::size_t Count, typename Head, typename Tail>
            struct drop_impl<Count, data::cons_<Head, Tail>>
                    : drop_impl<Count - 1, util::data<Tail>> {};
            template<std::size_t Count>
            struct drop_impl<Count, data::nil_<>> : data::nil_<> {};
            template<typename Head, typename Tail>
            struct drop_impl<0, data::cons_<Head, Tail>> : data::cons_<Head, Tail> {};
        } // namespace detail
        struct drop {
            template<typename Count, typename List>
            struct apply : detail::drop_impl<util::eval<Count>::value, util::data<List>> {};
        };
        template<typename Count, typename List>
        using drop_ = apply_<drop, Count, List>;

        // take_while: take elements from beginning of list
        struct take_while {
            template<typename Pred, typename List>
            struct apply : apply<Pred, util::data<List>> {};
            template<typename Pred>
            struct apply<Pred, data::nil_<>> : data::nil_<> {};
            template<typename Pred, typename Head, typename Tail>
            struct apply<Pred, data::cons_<Head, Tail>>
                    : cond_<apply_<Pred, Head>,
                              data::cons_<Head, apply<Pred, Tail>>,
                              data::nil_<>> {};
        };
        template<typename Pred, typename List>
        using take_while_ = apply_<take_while, Pred, List>;

        // drop_while: drop elements from beginning of list
        struct drop_while {
            template<typename Pred, typename List>
            struct apply : apply<Pred, util::data<List>> {};
            template<typename Pred>
            struct apply<Pred, data::nil_<>> : data::nil_<> {};
            template<typename Pred, typename Head, typename Tail>
            struct apply<Pred, data::cons_<Head, Tail>>
                    : cond_<apply_<Pred, Head>, apply<Pred, Tail>, data::cons_<Head, Tail>> {};
        };
        template<typename Pred, typename List>
        using drop_while_ = apply_<drop_while, Pred, List>;

        // index_of: get index of element in list
        struct index_of {
            template<typename Val, typename List>
            struct apply : cond_<equal_<Val, head_<List>>,
                                   util::val_size<0>,
                                   add_<util::val_size<1>, apply<Val, tail_<List>>>> {};
        };
        template<typename Val, typename List>
        using index_of_ = apply_<index_of, Val, List>;

        // zip_with: zip two lists and map a function over the result
        struct zip_with {
            template<typename Func, typename ListA, typename ListB>
            struct apply : apply<Func, util::data<ListA>, util::data<ListB>> {};
            template<typename Func>
            struct apply<Func, data::nil_<>, data::nil_<>> : data::nil_<> {};
            template<typename Func, typename ListA>
            struct apply<Func, ListA, data::nil_<>> : data::nil_<> {};
            template<typename Func, typename ListB>
            struct apply<Func, data::nil_<>, ListB> : data::nil_<> {};
            template<typename Func, typename HeadA, typename TailA, typename HeadB, typename TailB>
            struct apply<Func, data::cons_<HeadA, TailA>, data::cons_<HeadB, TailB>>
                    : data::cons_<apply_<Func, HeadA, HeadB>, apply<Func, TailA, TailB>> {};
        };
        template<typename Func, typename ListA, typename ListB>
        using zip_with_ = apply_<zip_with, Func, ListA, ListB>;

        // max: get maximum of list
        using max = fix_arg_count_<partial_<max_by, greater>, util::val_size<1>>;
        template<typename List>
        using max_ = apply_<max, List>;

        // min: get minimum of list
        using min = fix_arg_count_<partial_<max_by, less>, util::val_size<1>>;
        template<typename List>
        using min_ = apply_<min, List>;

        // all: and all values of a list
        using all = fix_arg_count_<partial_<fold_right, logic_and, util::val_bool<true>>,
                util::val_size<1>>;
        template<typename List>
        using all_ = apply_<all, List>;

        // any: or all values of a list
        using any = fix_arg_count_<partial_<fold_right, logic_or, util::val_bool<false>>,
                util::val_size<1>>;
        template<typename List>
        using any_ = apply_<any, List>;

        // sum: add all values of a list
        using sum = fix_arg_count_<partial_<fold_right, add, util::val_int<0>>, util::val_size<1>>;
        template<typename List>
        using sum_ = apply_<sum, List>;

        // count: get length of list
        using count = compose_<sum, partial_<map, constant_<util::val_size<1>>>>;
        template<typename List>
        using count_ = apply_<count, List>;

        // count_if: get number of values satisfying predicate
        template<typename Pred, typename List>
        using count_if_ = count_<filter_<Pred, List>>;
        using count_if = util::func_from_templ<count_if_, 2>;

        // contains: test if list contains value
        template<typename Val, typename List>
        using contains_ = any_<map_<util::lambda<equal_<Val, util::arg<0>>>, List>>;
        using contains = util::func_from_templ<contains_, 2>;

        // partition: partition list according to predicate
        template<typename Pred, typename List>
        using partition_ =
                data::pair_<filter_<Pred, List>, filter_<compose_<logic_not, Pred>, List>>;
        using partition = util::func_from_templ<partition_, 2>;

        // sort_by: sort list by comparison function
        struct sort_by {
            template<typename Cmp, typename List>
            struct apply : apply<Cmp, util::data<List>> {};
            template<typename Cmp>
            struct apply<Cmp, data::nil_<>> : data::nil_<> {};
            template<typename Cmp, typename Head, typename Tail>
            struct apply<Cmp, data::cons_<Head, Tail>>
                    : apply_<util::lambda<join_<make_list_<first_<util::arg<0>>,
                                     make_list_<Head>,
                                     second_<util::arg<0>>>>>,
                              map_<partial_<sort_by, Cmp>, partition_<partial_<Cmp, Head>, Tail>>> {
            };
        };
        template<typename Cmp, typename List>
        using sort_by_ = apply_<sort_by, Cmp, List>;

        // sort: sort list ascending
        using sort = fix_arg_count_<partial_<sort_by, greater>, util::val_size<1>>;
        template<typename List>
        using sort_ = apply_<sort, List>;

    } // namespace func

    namespace util {
        // utilities which depend on some of the meta-functions

        // lists of values
        template<typename T, T... Vals>
        using vals = func::make_list_<val<T, Vals>...>;
        template<std::size_t... Vals>
        using vals_size = vals<std::size_t, Vals...>;
        template<int... Vals>
        using vals_int = vals<int, Vals...>;
        template<bool... Vals>
        using vals_bool = vals<bool, Vals...>;

    } // namespace util

    namespace data {
        // operations on data types are implemented here

        // ---------------- type classes ----------------

        template<typename Self>
        template<typename MonadMonad>
        struct monad<Self>::join::apply : func::bind_<MonadMonad, func::id> {};

        template<typename Self>
        template<typename Monad, typename Func>
        struct monad<Self>::bind::apply : func::join_<func::map_<Func, Monad>> {};

        // ---------------- list ----------------

        template<typename Func, typename List>
        struct list::map::apply : list::map::apply<Func, util::data<List>> {};
        template<typename Func>
        struct list::map::apply<Func, nil_<>> : nil_<> {};
        template<typename Func, typename Head, typename Tail>
        struct list::map::apply<Func, cons_<Head, Tail>>
                : cons_<func::apply_<Func, Head>, list::map::apply<Func, Tail>> {};

        template<typename Val>
        struct list::unit::apply : cons_<Val, nil_<>> {};

        template<typename ListOfLists>
        struct list::join::apply : func::fold_right_<func::concat, nil_<>, ListOfLists> {};

        // ---------------- maybe ----------------

        template<typename Func, typename Maybe>
        struct maybe::map::apply : maybe::map::apply<Func, util::data<Maybe>> {};
        template<typename Func>
        struct maybe::map::apply<Func, nothing_<>> : nothing_<> {};
        template<typename Func, typename Val>
        struct maybe::map::apply<Func, just_<Val>> : just_<func::apply_<Func, Val>> {};

        template<typename Val>
        struct maybe::unit::apply : just_<Val> {};

        template<typename MaybeMaybe>
        struct maybe::join::apply : maybe::join::apply<util::data<MaybeMaybe>> {};
        template<>
        struct maybe::join::apply<nothing_<>> : nothing_<> {};
        template<typename Maybe>
        struct maybe::join::apply<just_<Maybe>> : Maybe {};

        // ---------------- pair ----------------

        template<typename Func, typename Functor>
        struct pair_type::map::apply : pair_type::map::apply<Func, util::data<Functor>> {};
        template<typename Func, typename First, typename Second>
        struct pair_type::map::apply<Func, pair_<First, Second>>
                : pair_<func::apply_<Func, First>, func::apply_<Func, Second>> {};
    } // namespace data

    // includes all other namespaces for convenience
    namespace all {

        using namespace data;
        using namespace util;
        using namespace func;

        using namespace args;
        using namespace vars;

    } // namespace all

    // tests to verify functions work somewhat
    namespace test {

        using namespace all;

        template<typename T>
        struct print;

        // eval, uneval
        static_assert(std::is_same_v<int, eval<int>>);
        static_assert(std::is_same_v<int, eval<uneval<int>>>);
        static_assert(std::is_same_v<int, eval<uneval<uneval<int>>>>);
        static_assert(std::is_same_v<int, eval<eval<int>>>);

        // func_from_templ
        static_assert(
                std::is_same_v<int *, eval<apply_<func_from_templ<std::add_pointer_t>, int>>>);
        static_assert(eval<arg_count_<func_from_templ<std::is_same, 2>>>::value == 2);

        // lambda
        static_assert(eval<at_<val_size<1>,
                              map_<lambda<apply_<func_from_templ<std::is_pointer>, _1>>,
                                      make_list_<short *, int, long>>>>::value
                == false);
        static_assert(std::is_same_v<eval<make_list_<val_int<2>, val_int<3>, val_int<4>>>,
                eval<apply_<lambda<map_<lambda<add_<val_int<1>, arg<0>>>,
                                    make_list_<arg<0>, val_int<2>, val_int<3>>>>,
                        val_int<1>>>>);

        // lambda2
        static_assert(eval<at_<val_size<1>,
                              map_<lambda2<a, apply_<func_from_templ<std::is_pointer>, a>>,
                                      make_list_<short *, int, long>>>>::value
                == false);
        static_assert(std::is_same_v<eval<make_list_<val_int<2>, val_int<3>, val_int<4>>>,
                eval<apply_<lambda2<var<0>,
                                    map_<lambda2<var<0>, add_<val_int<1>, var<0>>>,
                                            make_list_<var<0>, val_int<2>, val_int<3>>>>,
                        val_int<1>>>>);
        static_assert(std::is_same_v<eval<make_list_<val_int<2>, val_int<3>, val_int<4>>>,
                eval<apply_<lambda2<var<0>,
                                    map_<lambda2<var<1>, add_<var<0>, var<1>>>,
                                            make_list_<var<0>, val_int<2>, val_int<3>>>>,
                        val_int<1>>>>);
        static_assert(std::is_same_v<eval<make_list_<val_int<2>, val_int<3>, val_int<4>>>,
                eval<apply_<lambda2<var<0>,
                                    map_<lambda<add_<var<0>, arg<0>>>,
                                            make_list_<var<0>, val_int<2>, val_int<3>>>>,
                        val_int<1>>>>);
        static_assert(std::is_same_v<eval<make_list_<val_int<2>, val_int<3>, val_int<4>>>,
                eval<apply_<lambda<map_<lambda2<var<1>, add_<arg<0>, var<1>>>,
                                    make_list_<arg<0>, val_int<2>, val_int<3>>>>,
                        val_int<1>>>>);

        // map
        static_assert(std::is_same_v<nil_<>, eval<map_<id, nil_<>>>>);
        static_assert(std::is_same_v<just_<int>, eval<map_<id, just_<int>>>>);

        // join
        static_assert(std::is_same_v<int, eval<head_<join_<cons_<cons_<int, nil_<>>, nil_<>>>>>>);

        // bind
        static_assert(std::is_same_v<int, eval<int>>);

        // head
        static_assert(std::is_same_v<int, eval<head_<cons_<int, nil_<>>>>>);

        // tail
        static_assert(std::is_same_v<int, eval<head_<tail_<cons_<int, cons_<int, nil_<>>>>>>>);

        // fix_arg_count
        static_assert(eval<arg_count_<fix_arg_count_<add, val_size<2>>>>::value == 2);

        // apply, id
        static_assert(std::is_same_v<int, eval<apply_<id, int>>>);

        // constant
        static_assert(std::is_same_v<int, eval<apply_<constant_<int>>>>);
        static_assert(std::is_same_v<int, eval<apply_<constant_<int>, short, long>>>);

        // make_list
        static_assert(
                std::is_same_v<cons_<int, cons_<short, nil_<>>>, eval<make_list_<int, short>>>);

        // add, sub, mul, div
        static_assert(eval<add_<val_int<1>, val_int<2>, val_int<3>>>::value == 6);
        static_assert(eval<sub_<val_int<10>, val_int<13>>>::value == -3);
        static_assert(eval<mul_<val_int<1>, val_int<3>, val_int<3>>>::value == 9);
        static_assert(eval<div_<val_int<10>, val_int<3>>>::value == 3);

        // greater
        static_assert(eval<greater_<val_int<2>, val_int<1>>>::value);

        // less
        static_assert(eval<less_<val_int<1>, val_int<2>>>::value);

        // equal
        static_assert(eval<equal_<val_int<2>, val_int<2>>>::value);

        // greater_equal
        static_assert(eval<greater_equal_<val_int<2>, val_int<1>>>::value);

        // less_equal
        static_assert(eval<less_equal_<val_int<1>, val_int<2>>>::value);

        // logic_and
        static_assert(eval<logic_and_<val_bool<true>, val_bool<true>>>::value);

        // logic_or
        static_assert(eval<logic_or_<val_bool<false>, val_bool<true>>>::value);

        // logic_not
        static_assert(eval<logic_not_<val_bool<false>>>::value);

        // max_by
        static_assert(eval<max_by_<greater, make_list_<val_int<1>, val_int<3>, val_int<-2>>>>::value
                == 3);

        // iterate
        static_assert(std::is_same_v<int **,
                eval<head_<tail_<tail_<iterate_<func_from_templ<std::add_pointer_t>, int>>>>>>);

        // repeat
        static_assert(eval<arg_count_<repeat>>::value == 1);
        static_assert(std::is_same_v<int, eval<head_<tail_<tail_<repeat_<int>>>>>>);
        static_assert(std::is_same_v<int, eval<head_<tail_<tail_<map_<id, repeat_<int>>>>>>>);

        // concat
        static_assert(std::is_same_v<eval<make_list_<int, short>>,
                eval<concat_<cons_<int, nil_<>>, cons_<short, nil_<>>>>>);

        // fold_left
        static_assert(eval<fold_left_<add, val_int<10>, make_list_<val_int<1>, val_int<4>>>>::value
                == 15);

        // fold_right
        static_assert(eval<fold_right_<add, val_int<10>, make_list_<val_int<1>, val_int<4>>>>::value
                == 15);

        // filter
        static_assert(std::is_same_v<eval<make_list_<int *, short *>>,
                eval<filter_<func_from_templ<std::is_pointer>,
                        make_list_<int, int *, short, short *>>>>);
        static_assert(std::is_same_v<int *,
                eval<head_<filter_<func_from_templ<std::is_pointer>, repeat_<int *>>>>>);

        // at
        static_assert(std::is_same_v<int, eval<at_<val_int<1>, make_list_<short, int, long>>>>);

        // append
        static_assert(std::is_same_v<eval<make_list_<short, long, int>>,
                eval<append_<int, make_list_<short, long>>>>);

        // prepend
        static_assert(std::is_same_v<eval<make_list_<int, short, long>>,
                eval<prepend_<int, make_list_<short, long>>>>);

        // reverse
        static_assert(std::is_same_v<eval<make_list_<short, int, long>>,
                eval<reverse_<make_list_<long, int, short>>>>);

        // take
        static_assert(std::is_same_v<eval<make_list_<>>,
                eval<take_<val_int<0>, make_list_<int, short, long>>>>);
        static_assert(std::is_same_v<eval<make_list_<int, short>>,
                eval<take_<val_int<2>, make_list_<int, short, long>>>>);
        static_assert(std::is_same_v<eval<make_list_<int, short, long>>,
                eval<take_<val_int<3>, make_list_<int, short, long>>>>);
        static_assert(std::is_same_v<eval<make_list_<int, short, long>>,
                eval<take_<val_int<4>, make_list_<int, short, long>>>>);

        // drop
        static_assert(std::is_same_v<eval<make_list_<int, short, long>>,
                eval<drop_<val_int<0>, make_list_<int, short, long>>>>);
        static_assert(std::is_same_v<eval<make_list_<long>>,
                eval<drop_<val_int<2>, make_list_<int, short, long>>>>);
        static_assert(std::is_same_v<eval<make_list_<>>,
                eval<drop_<val_int<3>, make_list_<int, short, long>>>>);
        static_assert(std::is_same_v<eval<make_list_<>>,
                eval<drop_<val_int<4>, make_list_<int, short, long>>>>);

        // take_while
        static_assert(std::is_same_v<eval<make_list_<int *, short *>>,
                eval<take_while_<func_from_templ<std::is_pointer>,
                        make_list_<int *, short *, int, long *>>>>);
        static_assert(std::is_same_v<eval<make_list_<int *, short *>>,
                eval<take_while_<func_from_templ<std::is_pointer>, make_list_<int *, short *>>>>);
        static_assert(std::is_same_v<eval<make_list_<>>,
                eval<take_while_<func_from_templ<std::is_pointer>, make_list_<int, short *>>>>);

        // drop_while
        static_assert(std::is_same_v<eval<make_list_<int, long *>>,
                eval<drop_while_<func_from_templ<std::is_pointer>,
                        make_list_<int *, short *, int, long *>>>>);
        static_assert(std::is_same_v<eval<make_list_<>>,
                eval<drop_while_<func_from_templ<std::is_pointer>, make_list_<int *, short *>>>>);
        static_assert(std::is_same_v<eval<make_list_<int, short *>>,
                eval<drop_while_<func_from_templ<std::is_pointer>, make_list_<int, short *>>>>);

        // arg_count
        static_assert(eval<arg_count_<sub>>::value == 2);
        static_assert(eval<arg_count_<add>>::value == 0);
        static_assert(eval<arg_count_<lambda<int>>>::value == 0);
        static_assert(eval<arg_count_<lambda2<a, b, int>>>::value == 2);
        static_assert(eval<arg_count_<count>>::value == 1);

        // curry_n
        static_assert(
                eval<apply_<apply_<curry_n_<add, val_size<2>>, val_int<10>>, val_int<1>>>::value
                == 11);
        static_assert(eval<apply_<apply_<curry_n_<add, val_int<3>>, val_int<1>, val_int<2>>,
                              val_int<3>>>::value
                == 6);

        // curry
        static_assert(eval<apply_<apply_<curry_<sub>, val_int<10>>, val_int<1>>>::value == 9);
        static_assert(eval<apply_<curry_<sub>, val_int<10>, val_int<1>>>::value == 9);

        // compose
        using add_c = curry_n_<add, val_int<2>>;
        using inc = apply_<add_c, val_int<1>>;
        using dec = apply_<add_c, val_int<-1>>;
        static_assert(eval<apply_<compose_<>, val_int<5>>>::value == 5);
        static_assert(eval<apply_<compose_<inc>, val_int<5>>>::value == 6);
        static_assert(eval<apply_<compose_<inc, dec, dec>, val_int<5>>>::value == 4);

        // partial
        static_assert(eval<apply_<partial_<add, val_int<1>, val_int<2>>, val_int<3>>>::value == 6);

        // y_comb
        using fac = y_comb_<lambda2<f,
                x,
                n,
                cond_<equal_<n, val_int<0>>,
                        val_int<1>,
                        mul_<n, apply_<f, void, sub_<n, val_int<1>>>>>>>;
        static_assert(std::is_same_v<eval<vals_int<1, 1, 6>>,
                eval<map_<partial_<fac, int>, vals_int<0, 1, 3>>>>);

        // max
        static_assert(eval<max_<make_list_<val_int<1>, val_int<3>, val_int<-2>>>>::value == 3);

        // min
        static_assert(eval<min_<make_list_<val_int<1>, val_int<3>, val_int<-2>>>>::value == -2);

        // all
        static_assert(!eval<all_<repeat_<val_bool<false>>>>::value);

        // any
        static_assert(eval<any_<repeat_<val_bool<true>>>>::value);

        // count
        static_assert(eval<count_<make_list_<int, short, long>>>::value == 3);

        // count_if
        static_assert(eval<count_if_<func_from_templ<std::is_pointer>,
                              make_list_<int, short *, long>>>::value
                == 1);
        static_assert(eval<apply_<count_if,
                              func_from_templ<std::is_pointer>,
                              make_list_<int, short *, long *>>>::value
                == 2);

        // contains
        static_assert(eval<contains_<int, repeat_<int>>>::value);

        // partition
        static_assert(std::is_same_v<
                eval<pair_<make_list_<int *, short *, long *>, make_list_<short, int, long>>>,
                eval<partition_<func_from_templ<std::is_pointer>,
                        make_list_<short, int, int *, short *, long, long *>>>>);

        // index_of
        static_assert(eval<index_of_<int, make_list_<short, int, long>>>::value == 1);

        // sort
        static_assert(
                std::is_same_v<eval<make_list_<val_int<-2>, val_int<0>, val_int<1>, val_int<3>>>,
                        eval<sort_<make_list_<val_int<1>, val_int<0>, val_int<-2>, val_int<3>>>>>);

        // sort_by
        static_assert(std::is_same_v<
                eval<make_list_<val_int<3>, val_int<1>, val_int<0>, val_int<-2>>>,
                eval<sort_by_<less, make_list_<val_int<1>, val_int<0>, val_int<-2>, val_int<3>>>>>);

        // zip_with
        static_assert(std::is_same_v<eval<vals_int<2, 6>>,
                eval<zip_with_<mul, vals_int<2, 3>, vals_int<1, 2>>>>);

    } // namespace test

} // namespace metafunc
