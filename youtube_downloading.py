from pytube import YouTube

def download_video(url, save_path='.'):
    try:
        yt = YouTube(url)
        print("Downloading...")
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=save_path)
        print(f"Downloaded: {yt.title}")
    except Exception as e:
        print(f"Error: {e}")

def download_playlist(playlist_url, save_path='.'):
    try:
        playlist = Playlist(playlist_url)
        print(f'Downloading playlist: {playlist.title}')
        for video in playlist.videos:
            video.streams.get_highest_resolution().download(output_path=save_path)
            print(f'Downloaded: {video.title}')
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    with open('videos-youtube.txt',encoding='utf-8') as f:
        urls = f.readlines()
        print(f"Total {len(urls)} urls:")
        for url in urls:
            video_url = url  # Example video URL
            download_video(video_url, save_path='D:\Download\RNA-2024-meeting')

