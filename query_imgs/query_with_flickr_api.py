from flickrapi import FlickrAPI

FLICKR_PUBLIC = 'f8b9cd454275fca40682a0e602c67073'
FLICKR_SECRET = 'cab676a88194ebae'

flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')
extras='url_sq,url_t,url_s,url_q,url_m,url_n,url_z,url_c,url_l,url_o'
cats = flickr.photos.search(text='kitten', per_page=5, extras=extras)
photos = cats['photos']
from pprint import pprint
pprint(photos)