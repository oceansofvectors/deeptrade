from warcio.archiveiterator import ArchiveIterator
with open('segment.warc.gz', 'rb') as stream:
    for record in ArchiveIterator(stream):
        if record.rec_type == 'response':
            raw_content = record.content_stream().read()  # This is the raw HTTP response
            print(raw_content.decode('utf-8'))  # Decode if needed for viewing
