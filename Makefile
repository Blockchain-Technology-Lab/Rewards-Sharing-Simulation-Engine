# Define shell to use
SHELL := /bin/bash

# Define make commands
.PHONY: up down

# Command to build and start the container
up:
	@docker-compose up --build

# Command to stop and remove the container
down:
	@docker-compose down
