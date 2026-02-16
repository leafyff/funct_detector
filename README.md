## Development logs

### First plan: (09.01.2026)

- Відображення вікна (Qt)
- Мають бути інструменти для комфортного малювання
- Згладжування намальованого
- Зчитування та оцінка всіх варіантів функцій
- Генерація LateX коду на основі зчитаного (TikZ)


### v0.1 (21.01.2026)

Реалізовано наступні кроки:
- Відображення вікна (Qt)
- Згладжування намальованого
- Зчитування функції:
	- Поліноміальна апроксимація, 
	- Кубічні сплайни
	- Апроксимація базовими функціями: експоненціальна, логарифмічна, синусоїдальна
	- Оцінка ефективності: RMSE
- Генерація LateX коду на основі зчитаного

### v0.2 (12.02.2026)

Тепер доступні нові методи аналізу функцій:
- Polynomial Approximation (Interpolation) (Chebyshev Basis)
- Polynomial Least Squares (Chebyshev Basis)
- Cubic Splines  
- Non-Uniform Fast Fourier Transform (NUFFT)
- AAA Algorithm (Adaptive Antoulas-Anderson) (Nakatsukasa–Sète–Trefethen)
- Discrete Minimax via Linear Programming
- Simple functions: exponential, logarithmic, hyperbolic, sinusoidal, tangentoidal
Нові метрики аналізу якості апроксимації:
- RMSE - Корінь з середньоквадратичної похибки
- L-infinity / Minimax Error - Похибка за найгіршим значенням
- BIC (Bayesian Information Criterion)
- Multi-objective Score - Об'єднаний критерій для трьох вище, щоб дати єдину оцінку


### Можливо буде реалізовано (long-term plans):
- Генерація напряму TikZ коду 
- Графічний інтерфейс напряму в GitHub (Веб Дизайн)
- Аналіз параметричних кривих
- ШІ аналіз?
