# Collaborative-filter-system-recomendation
Calculos para generar recomendaciones a clientes segun comportamiento similar de otros clientes.
Para un uso de memoria y tiempo eficiente, se utiliza estrucutra de datos disperso y utilizacion de multiples hilos para reducir el tiempo y costo de memoria.

## Requerimientos

- [Python >= 3.6](https://www.python.org/)
- [Pipenv](https://github.com/pypa/pipenv)

## Instalación

1. Clone repository: `git clone https://github.com/Joaquinecc/api-bristol.git`
2. Install dependencies: `pipenv install`
3. Activate virtualenv: `pipenv shell`
4. Create a file called `settings-params.json` in root directory
5. Insert the following lines into the file:

```
{
    "user":<USER_DATABASE>,
    "password":<USER_DATABASE>,
    "host":<HOST_DATABASE>,
    "port":<PORT_DATABASE>,
    "database":<DATABASE_NAME>,
    "topNProduct":15,
    "topNSimilarity":30
}
```
6. Run script: `python main.py`

## Guia de uso


**topNProduct**: Cuantos recomendaciones generar para cada cliente

**topNSimilarity**: Con cuantos clientes se hace la comparación. (Se eligen primero los de mayor similitud)

