import glob
import os
import json
import time

if __name__ == "__main__":
    
    json_files = glob.glob("./meta/*json")
    print("# of json_files (# of painters):", len(json_files))
    time.sleep(1)

    total_art_counter = 0
    total_art_with_tags_counter = 0
    total_art_with_genre_counter = 0
    total_art_with_style_counter = 0
    total_art_with_material_counter = 0
    total_art_with_serie_counter = 0
    total_art_with_technique_counter = 0
    total_art_with_location_counter = 0

    for json_file in json_files:

        if os.path.basename(json_file) == 'artists.json':
            continue

        with open(json_file, 'r') as f:
            json_obj = json.load(f)

        print(json_file)
        print("\t# of arts:", len(json_obj))

        tag_counter = 0
        genre_counter = 0
        style_counter = 0
        material_counter = 0
        serie_counter = 0
        technique_counter = 0
        location_counter = 0
        for art in json_obj:
            total_art_counter += 1

            if art.get('tags'):
                tag_counter += 1
                total_art_with_tags_counter += 1

            if art.get('genre'):
                genre_counter += 1
                total_art_with_genre_counter += 1

            if art.get('style'):
                style_counter += 1
                total_art_with_style_counter += 1

            if art.get('material'):
                material_counter += 1
                total_art_with_material_counter += 1

            if art.get('serie'):
                serie_counter += 1
                total_art_with_serie_counter += 1
                print(art['serie'])

            if art.get('technique'):
                technique_counter += 1
                total_art_with_technique_counter += 1

            if art.get('location'):
                location_counter += 1
                total_art_with_location_counter += 1

        # print("\t# of arts with tags:", tag_counter)
        # print("\t# of arts with genre:", genre_counter)
        # print("\t# of arts with style:", style_counter)
        # print("\t# of arts with material:", material_counter)
        # print("\t# of arts with serie:", serie_counter)
        # print("\t# of arts with technique:", technique_counter)
        # print("\t# of arts with location:", location_counter)

    print("Total # of arts:", total_art_counter)
    print("Total # of arts with tags:", total_art_with_tags_counter)
    print("Total # of arts with genre:", total_art_with_genre_counter)
    print("Total # of arts with style:", total_art_with_style_counter)
    print("Total # of arts with material:", total_art_with_material_counter)
    print("Total # of arts with serie:", total_art_with_serie_counter)
    print("Total # of arts with technique:", total_art_with_technique_counter)
    print("Total # of arts with location:", total_art_with_location_counter)