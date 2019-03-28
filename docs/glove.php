<?php
header('Content-type: application/json');

$word = trim($_GET['word']);

$F = fopen("embeddings/glove.50d.embeddings.txt", "r");

if ($F) {
    while (($line = fgets($F)) !== false) {
        $c = explode(" ", trim($line));
        $c_word = array_shift($c);
        if ($c_word == $word) {            
            print(json_encode($c));
            break;
        }
    }

    fclose($F);
} else {
    die(0);
} 
