"""The serving lane: FastAPI over the model artifact written by the training lane.

Nothing in here trains, fits or mutates a model. The API loads `models/model.joblib` once at
startup and reads `models/model_card.json` for the grade cut-offs, so the numbers it serves
are the ones a real run produced rather than constants typed into serving code.
"""
