# Run all tests
docker run --rm -v "${PWD}:/app" kdsh-pipeline python -m pytest tests/ -v

# Run integration test
docker run --rm -v "${PWD}:/app" kdsh-pipeline python tests/test_integration.py

# Interactive shell
docker run -it --rm -v "${PWD}:/app" kdsh-pipeline bash

docker-compose run --rm pipeline python -m pytest tests/ -v -k "not slow"