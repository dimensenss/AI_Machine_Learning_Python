% Факт визначення ряду Фібоначчі
fib(0, 0).
fib(1, 1).

% Факт для знаходження числа ряду Фібоначчі
fib(N, Result) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, Result1),
    fib(N2, Result2),
    Result is Result1 + Result2.
% Правило для знаходження N-го числа ряду Фібоначчі
find_fibonacci(N, Result) :-
    fib(N, Result).
