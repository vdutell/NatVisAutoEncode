## helper functions for animation

##embed animation as html

from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import base64

import utils.plotutils as plu

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=30, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('utf-8')
    
    return VIDEO_TAG.format(anim._encoded_video)

from IPython.display import HTML


#function we can call to animate with HTML
def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

#vanilla animation function
def animater(i, im, clip):
    im.set_array(clip[i].ravel())
    return im,

def animate_weights_plotter(i, wer, imobj):
    img = plu.display_data_tiled(wer[i], normalize=False, 
                                title="weights_evolving", 
                                prev_fig=None)
    imobj.set_data(img)
    return imobj,