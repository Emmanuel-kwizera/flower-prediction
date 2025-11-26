from locust import HttpUser, task, between

class FlowerUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def root(self):
        self.client.get("/")
