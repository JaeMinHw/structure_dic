import yt_dlp

url = "https://www.youtube.com/watch?v=5yB6j65MtXg"

ydl_opts = {
    'outtmpl': '%(title)s.%(ext)s',  # 파일명
    'format': 'bestvideo+bestaudio/best',  # 최고 화질
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])