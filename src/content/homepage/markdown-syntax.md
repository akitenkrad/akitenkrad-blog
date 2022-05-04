+++
author = "akitenkrad"
title = "Markdown Syntax Guide"
date = "2022-05-03"
description = "マークダウンの表記法まとめ"
tags = [
    "markdown",
]
categories = [
    "themes",
    "syntax",
]
series = ["Themes Guide"]
+++

## ヘッダ

```
# H1
## H2
### H3
#### H4
##### H5
###### H6
```

# H1
## H2
### H3
#### H4
##### H5
###### H6

## パラグラフ

隴西の李徴は博学才穎、天宝の末年、若くして名を虎榜に連ね、ついで江南尉に補せられたが、性、狷介、自ら恃むところ頗る厚く、賤吏に甘んずるを潔しとしなかった。いくばくもなく官を退いた後は、故山、かく略かくに帰臥し、人と交を絶って、ひたすら詩作に耽った。下吏となって長く膝を俗悪な大官の前に屈するよりは、詩家としての名を死後百年に遺そうとしたのである。しかし、文名は容易に揚らず、生活は日を逐うて苦しくなる。李徴は漸く焦躁に駆られて来た。この頃ころからその容貌も峭刻となり、肉落ち骨秀で、眼光のみ徒に炯々として、曾て進士に登第した頃の豊頬の美少年の俤は、何処に求めようもない。数年の後、貧窮に堪たえず、妻子の衣食のために遂に節を屈して、再び東へ赴き、一地方官吏の職を奉ずることになった。

一方、これは、己の詩業に半ば絶望したためでもある。曾ての同輩は既に遥か高位に進み、彼が昔、鈍物として歯牙にもかけなかったその連中の下命を拝さねばならぬことが、往年の儁才李徴の自尊心を如何に傷けたかは、想像に難かたくない。

## バッククオート

```
`猛虎`
```

`猛虎`

## 引用

> 彼は怏々として楽しまず、狂悖の性は愈々抑え難がたくなった。

## テーブル

```
   Name | Age
--------|------
    Bob | 27
  Alice | 23
```

   Name | Age
--------|------
    Bob | 27
  Alice | 23

## インラインテーブル

```
| Inline&nbsp;&nbsp;&nbsp;     | Markdown&nbsp;&nbsp;&nbsp;  | In&nbsp;&nbsp;&nbsp;                | Table      |
| ---------- | --------- | ----------------- | ---------- |
| *italics*  | **bold**  | ~~strikethrough~~&nbsp;&nbsp;&nbsp; | `code`     |
```

| Inline&nbsp;&nbsp;&nbsp;     | Markdown&nbsp;&nbsp;&nbsp;  | In&nbsp;&nbsp;&nbsp;                | Table      |
| ---------- | --------- | ----------------- | ---------- |
| *italics*  | **bold**  | ~~strikethrough~~&nbsp;&nbsp;&nbsp; | `code`     |

## コードブロック

### バッククオート

    ```
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Example HTML5 Document</title>
    </head>
    <body></body>
    </html>
    ```

```
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Example HTML5 Document</title>
</head>
<body></body>
</html>
```

### インデント

```
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Example HTML5 Document</title>
    </head>
    <body></body>
    </html>
```

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Example HTML5 Document</title>
    </head>
    <body></body>
    </html>

## ハイライト

    { {< highlight html >} }
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Example HTML5 Document</title>
    </head>
    <body></body>
    </html>
    { {< /highlight >} }

{{< highlight html >}}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Example HTML5 Document</title>
</head>
<body></body>
</html>
{{< /highlight >}}

## リスト

#### 順序付きリスト

```
1. First item
2. Second item
3. Third item
```

1. First item
2. Second item
3. Third item

#### リスト

```
* First item
* Second item
* Third item
```

* First item
* Second item
* Third item

#### リストのネスト

```
* Item
  1. First Sub-item
  2. Second Sub-item
```

* Item
  1. First Sub-item
  2. Second Sub-item

## 数式

### インライン

```
$$ \varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887… $$
```

$ \varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887… $$

### ブロック

```
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$
```

$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } } 
$$
