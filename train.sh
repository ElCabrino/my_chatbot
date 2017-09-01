if [ $1 == '1' ]
then
	python3 exec.py --use_lstm=True --size=1024 --num_layers=1 --num_attns=1 --num_attns_output=0 --steps_per_checkpoint=50 —model= 'tmp/1attns_in_0attns_outLSTM'
elif [ $1 == '2' ] 
then
	python3 exec.py --use_lstm=True --size=1024 --num_layers=1 --num_attns=2 --num_attns_output=0 --steps_per_checkpoint=50 —model= 'tmp/2attns_in_0attns_outLSTM'
elif [ $1 == '3' ]
then
	python3 exec.py --use_lstm=False --size=1024 --num_layers=1 --num_attns=1 --num_attns_output=0 --steps_per_checkpoint=50 —model= 'tmp/1attns_in_0attns_outGRU'
elif [ $1 == '4' ]
then
	python3 exec.py --use_lstm=False --size=1024 --num_layers=1 --num_attns=2 --num_attns_output=0 --steps_per_checkpoint=50 —model= 'tmp/2attns_in_0attns_outGRU'
elif [ $1 == '5' ]
then
	python3 exec.py --use_lstm=True --size=1024 --num_layers=1 --num_attns=1 --num_attns_output=1 --steps_per_checkpoint=50 —model= 'tmp/1attns_in_1attns_outLSTM'
elif [ $1 == '6' ]
then
	python3 exec.py --use_lstm=True --size=1024 --num_layers=1 --num_attns=2 --num_attns_output=1 --steps_per_checkpoint=50 —model= 'tmp/2attns_in_1attns_outLSTM'		
else
	echo 'bad argument'
fi