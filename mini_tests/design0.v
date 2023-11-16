module t(input [3: 0] i1, input [3: 0] i2, output [3: 0] o);
 assign o[0] = ^i2[3:0] && i1[0] >> i2[0] ;
 assign o[1] = ~|i2[2:1] >> !i1[1] >> !i1[1] ;
 assign o[2] = ~|i2[2:0] + i2[0] || !i1[2] ;
 assign o[3] = ~|i1[1:0] / !i2[2] + i1[1] ;
 endmodule