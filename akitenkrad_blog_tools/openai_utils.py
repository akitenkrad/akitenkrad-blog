import os
import time

import openai


def call_openai_chatcompletion(messages: list[dict[str, str]], model: str = "gpt-3.5-turbo-16k", retry_count: int = 5):
    while retry_count > 0:
        try:
            res = openai.ChatCompletion.create(
                model=model,
                messages=messages,
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
