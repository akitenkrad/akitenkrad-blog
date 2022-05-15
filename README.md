![GitHub Workflow Status](https://img.shields.io/github/workflow/status/akitenkrad/blog-akitenkrad/github%20pages?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/akitenkrad/blog-akitenkrad?style=for-the-badge)

# Akitenkrad's Blog

## Environment

1. Start Docker  
    ```bash
    $ docker-compose up -d
    $
    ```
2. Enter the container  
    ```bash
    $ docker-compose exec python /bin/bash
    ```
3. Create python environment with pipenv and enter the environment  
    ```bash
    $ pipenv install --python 3.9 --dev
    $ pipenv shell
    ```
4. Up Hugo development server  
    ```bash
    $ cd src
    $ hugo server --bind 0.0.0.0
    ```
5. Access local development Hugo site  
    -> http://localhost:1313/blog-akitenkrad/

## Generate New Post

```bash
# create index.md
$ python -m cli new-post
New Post -> /workplace/src/content/posts/papers/yyyyMMddHHMMSS/index.md

# collect references for a "paper" post
$ python -m cli get-references --title "<TITLE OF THE PAPER>" >> /workplace/src/content/posts/papers/yyyyMMddHHMMSS/index.md
```

replace hero.jpg with the image you would like
