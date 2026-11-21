BASE ?= ghcr.io/home-assistant/amd64-base-debian:trixie-2026.11.0
SMOKE_PORT ?= 10499

.PHONY: test image smoke

test:
	uv run pytest -q

image:
	docker build -f pypryon/Dockerfile --build-arg BUILD_FROM=$(BASE) -t pypryon-addon .

# run the image's server against your own model (PYPRYON_MODEL_DIR) and stream the espeak clip
smoke:
	@test -n "$(PYPRYON_MODEL_DIR)" || { echo "set PYPRYON_MODEL_DIR to an extracted ALEXA model dir"; exit 1; }
	docker rm -f pypryon-smoke >/dev/null 2>&1 || true
	docker run -d --name pypryon-smoke -p 127.0.0.1:$(SMOKE_PORT):10400 \
		-v "$(PYPRYON_MODEL_DIR)":/models/ALEXA:ro \
		--entrypoint /opt/venv/bin/pypryon-wyoming pypryon-addon --models-dir /models
	until docker logs pypryon-smoke 2>&1 | grep -q Ready; do sleep 0.5; \
		docker inspect -f '{{.State.Running}}' pypryon-smoke | grep -q true || { docker logs pypryon-smoke; exit 1; }; done
	uv run python tools/wyoming_client.py 127.0.0.1 $(SMOKE_PORT) tests/data/alexa_espeak.wav | tee .cache-smoke.txt
	docker logs pypryon-smoke 2>&1 | tail -3
	docker rm -f pypryon-smoke >/dev/null
	grep -q "detection alexa" .cache-smoke.txt && rm -f .cache-smoke.txt
