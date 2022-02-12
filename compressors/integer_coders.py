from core.data_block import DataBlock
from core.framework import DataEncoder, DataDecoder
from utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from utils.test_utils import try_lossless_compression


class UniversalUintEncoder(DataEncoder):
    """
    Universal Encoding:
    0 -> 100
    1 -> 101
    2 -> 11010
    3 -> 11011
    4 -> 1110100 (1110 + 100)
    ...
    NOTE: not the most efficient but still "universal"
    """

    def encode_symbol(self, x: int):
        assert isinstance(x, int)
        assert x >= 0

        symbol_bitarray = uint_to_bitarray(x)
        len_bitarray = BitArray(len(symbol_bitarray) * "1" + "0")
        return len_bitarray + symbol_bitarray

    def encode_block(self, data_block: DataBlock):
        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class UniversalUintDecoder(DataDecoder):
    """
    Universal Encoding:
    0 -> 100
    1 -> 101
    2 -> 11010
    3 -> 11011
    4 -> 1110100 (1110 + 100)
    ...
    NOTE: not the most efficient but still "universal"
    """

    def decode_symbol(self, encoded_bitarray):

        # initialize num_bits_consumed
        num_bits_consumed = 0

        # get the symbol length
        while True:
            bit = encoded_bitarray[num_bits_consumed]
            num_bits_consumed += 1
            if bit == 0:
                break
        num_ones = num_bits_consumed - 1

        # decode the symbol
        symbol = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + num_ones]
        )
        num_bits_consumed += num_ones

        return symbol, num_bits_consumed

    def decode_block(self, bitarray: BitArray):
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed


def test_universal_uint_encode_decode():
    encoder = UniversalUintEncoder()
    decoder = UniversalUintDecoder()

    # create some sample data
    data_list = [0, 0, 1, 3, 4, 100]
    data_block = DataBlock(data_list)

    is_lossless, _ = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless
