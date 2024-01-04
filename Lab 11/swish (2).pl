% Власна реалізація конкатенації списків
my_append([], L, L). % конкатенація порожнього списку з будь-яким списком дає той самий список
my_append([H|T], L, [H|R]) :- my_append(T, L, R). % Рекурсивно додає голову до результату конкатенації залишку списку та другого списку

% Власна реалізація порівняння двох елементів
my_dif(X, Y) :- X \= Y. % Визначає, чи два елементи неоднакові

% Предикат для видалення послідовності зі списку
remove_sequence([], _, []). %видалення з порожнього списку дає порожній список

% Якщо поточна частина списку починається з заданої послідовності, пропустити її
remove_sequence(List, Sequence, Result) :-
    my_append(Sequence, Rest, List), % Застосовує функцію конкатенації списків для отримання залишку
    remove_sequence(Rest, Sequence, Result). % Рекурсивно викликає саму себе для обробки залишку

% Якщо поточна частина списку не починається з заданої послідовності, додати голову до результату і продовжити рекурсію
remove_sequence([Head|Tail], Sequence, [Head|Result]) :-
    my_dif(Head, Sequence), % Використовуємо функцію порівняння елементів для визначення, чи голова не співпадає з послідовністю
    remove_sequence(Tail, Sequence, Result). % Рекурсивний виклик для обробки залишку та формування результату