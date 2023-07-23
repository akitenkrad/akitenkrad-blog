import os
import time

import openai

openai.organization = "org-gIuy12buVy2ONIQTiubeuq4a"
openai.api_key = os.environ.get("OPENAI_API_KEY")


def call_openai_chatcompletion(content: str, model: str = "gpt-3.5-turbo-16k", retry_count: int = 5):
    while retry_count > 0:
        try:
            res = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "research assistant", "content": content}],
                temperature=1.0,
                max_tokens=8000,
            )
            return res["choices"][0]["message"]["content"]
        except openai.error.ServiceUnavailableError:
            retry_count -= 1
            time.sleep(5.0)
            continue
        except Exception as ex:
            raise ex

    return ""
