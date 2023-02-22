test:
	poetry run pytest
	poetry run isort --check ao3_disco_ranking tests --profile black
	poetry run black --check ao3_disco_ranking/ tests/
	poetry run mypy --config-file=mypy.ini ao3_disco_ranking/

fix-lint:
	isort ao3_disco_ranking/ tests/ --profile black
	black ao3_disco_ranking/ tests/
