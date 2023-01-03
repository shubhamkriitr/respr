import math
class NNUtils:
    
    def compute_output_shapes_and_stats(self, input_size: int, layer_info, rf0=1):
        """
        `layer_info`: list of kernel_size, dilation, padding and strides
        """
        in_shape = input_size # int
        out_shape = None
        r_before = rf0
        j_in = 1
        j_out = None
        out = []
        for layer_num, info in enumerate(layer_info, 1):
            k, d, p, s = info
            assert d > 0
            r = r_before + ((k-1)*d)*j_in
            j_out = j_in*s
            
            out_shape = math.floor((in_shape + 2*p - d*(k-1) - 1)/s + 1 )
            
            print(f"Layer[{str(layer_num).zfill(4)}] ({in_shape})->({out_shape}"
                  f") , receptive field: {r}")
            
            out.append((out_shape, r, j_out))
            # prep for next iter
            j_in = j_out
            r_before = r
            in_shape = out_shape
        
        return out
    
    def struct_to_flat(self, d):
        """This is only for a specific block structure.
        Creates a flat list of [kernel_size, dilation, padding and strides]
        which can be fed to `compute_output_shapes_and_stats`.
        """
        flat = []
        front = d["front"]
        ach = d["channel_adjust"]
        
        for i in range(len(front)):
            b_front = front[i]
            
            for j in range(b_front[0]):
                _, ch, ds, ps, ss, dropout = b_front
                k = 3 # kernel size
                # assert n == 2
                for i2 in range(2):
                    flat.append([k, ds[i2], ps[i2], ss[i2]])
                    
            
            if i < (len(front) - 1):
                b_ach = ach[i]
                in_ch, out_ch, ds, ps, ss, dropout_p = b_ach
                k = 3
                flat.append([k, ds[0], ps[0], ss[0]])
        return flat
        
        
        