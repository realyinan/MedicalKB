from locust import TaskSet, task, between
from locust.contrib.fasthttp import FastHttpUser
import json

class NerTaskSet(TaskSet):
    @task
    def test_slot_recognizer(self):
        url = "/service/api/medical_ner"
        data = {"text": "我朋友的父亲除了患有糖尿病,无手术外伤史及药物过敏史"}
        headers = {'Content-Type': 'application/json;charset=utf8'}

        with self.client.post(url, data=json.dumps(data), headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response_data = json.loads(response.text)
                if "result" in response_data:
                    response.success()
                else:
                    response.failure("No 'result' key in response")
            else:
                response.failure(f"Request failed with status code: {response.status_code}")

class SlotRecognizerUser(FastHttpUser):
    wait_time = between(1, 3)
    tasks = [NerTaskSet]