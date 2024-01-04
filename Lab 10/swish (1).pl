% Рекурсивна функція для обчислення факторіалу
factorial(0, 1).
factorial(N, Result) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, Result1),
    Result is N * Result1.

% Рекурсивна функція для обчислення значення функції за рядом
approximate_function(N, X, Result) :-
    approximate_function(N, X, 0, Result).

approximate_function(0, _, Acc, Acc).
approximate_function(N, X, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, Factorial),
    Term is ((-1)^(N-1) * X^(2*N-2)) / Factorial,
    Acc1 is Acc + Term,
    approximate_function(N1, X, Acc1, Result).

% Головний предикат для виклику програми
main :-
    write('Enter the number of terms (N): '),
    read(N),
    write('Enter the value of X: '),
    read(X),
    
    % Обчислення наближеного значення функції
    approximate_function(N, X, ApproxResult),
    
    % Обчислення точного значення функції
    ExactResult is exp((-1) * X^2),
    
    % Виведення результатів на екран
    format('Approximate result: ~w~n', [ApproxResult]),
    format('Exact result: ~w~n', [ExactResult]).
