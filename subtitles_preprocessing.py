#https://youtu.be/a8hMgIcUEnE?si=yg1fQkHBUj6S7mB9
#https://kitsunekko.net/dirlist.php?dir=subtitles%2F
def delete_time_intervals(srt_file):
    with open(srt_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for i in range(len(lines)):
        if '-->' in lines[i]:
            # Supprimer la ligne avec le numéro de ligne
            new_lines.pop()
        elif lines[i].strip():  # Vérifier si la ligne n'est pas vide
            new_lines.append(lines[i])

    with open(srt_file, 'w') as f:
        f.writelines(new_lines)


delete_time_intervals('subtitles_test.srt')
