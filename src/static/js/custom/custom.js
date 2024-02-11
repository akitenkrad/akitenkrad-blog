import { Toast } from 'bootstrap'

// ------------------------------- start: copy to clipboard -------------------------------
function copyToClipboard(text, index) {
    return navigator.clipboard.writeText(text).then(function () {
        showCopyToClipBoardMessage(index)
    })
}

function showCopyToClipBoardMessage(index) {
    var toastElems = document.getElementsByClassName('toast-index-' + index)
    let toast = new Toast(toastElems[0], { autohide: true, delay: 2000, animation: true })
    toast.show()
}
// ------------------------------- end: copy to clipboard -------------------------------



// ------------------------------- start: file download ---------------------------------
function downloadFile(url, filename) {
    fetch(url).then(response => {
        response.blob().then(blob => {
            let url = window.URL.createObjectURL(blob)
            let a = document.createElement('a')
            a.href = url
            a.download = filename
            a.click()
        })
    })
}
// ------------------------------- end: file download------------------------------------

$(function () {
    // configure copy to clipboard buttons
    $('button.copy-to-clipboard').each(function () {
        $(this).click(function () {
            const title = $(this).attr("title")
            const index = $(this).attr("index")
            copyToClipboard(title, index)
        })
    })

    // configure file download buttons
    $('button.download-pdf').each(function () {
        $(this).click(function () {
            const url = $(this).attr("url")
            const filename = $(this).attr("filename")
            downloadFile(url, filename)
        })
    })
})