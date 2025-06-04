<?php
ini_set('memory_limit', '3024M');
ini_set('max_execution_time', 600);
error_reporting(1);
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
ini_set('log_errors', 1);
ini_set('error_log', 'cutting-error.log');
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

class GeneticCuttingOptimizer
{
    private $sheetWidth;
    private $sheetHeight;
    private $details = [];
    private $fitnessCache = [];

    // Параметры генетического алгоритма (значения по умолчанию)
    private $populationSize = 1000;
    private $generations    = 50;
    private $mutationRate   = 0.15;
    private $elitismRate    = 0.1;
    private $tournamentSize = 4;

    private $minScrapWidth = 0;
    private $minScrapHeight = 0;

    public function __construct($width, $height, $details, $options = [])
    {
        if (empty($details)) {
            throw new \Exception('Нет деталей для раскроя');
        }
        $this->sheetWidth  = (int)$width;
        $this->sheetHeight = (int)$height;

        // Применяем пользовательские настройки алгоритма
        if (isset($options['algorithmSettings'])) {
            $settings = $options['algorithmSettings'];

            if (isset($settings['populationSize'])) {
                $this->populationSize = (int)$settings['populationSize'];
            }
            if (isset($settings['generations'])) {
                $this->generations = (int)$settings['generations'];
            }
            if (isset($settings['mutationRate'])) {
                $this->mutationRate = (float)$settings['mutationRate'];
            }
            if (isset($settings['elitismRate'])) {
                $this->elitismRate = (float)$settings['elitismRate'];
            }
            if (isset($settings['tournamentSize'])) {
                $this->tournamentSize = (int)$settings['tournamentSize'];
            }
            if (isset($settings['minScrapWidth'])) {
                $this->minScrapWidth = (int)$settings['minScrapWidth'];
            }
            if (isset($settings['minScrapHeight'])) {
                $this->minScrapHeight = (int)$settings['minScrapHeight'];
            }
        }

        $index = 0;
        foreach ($details as $detail) {
            if (!isset($detail['width'], $detail['height'])) {
                throw new \Exception('Неверные данные детали');
            }
            $w = (int)$detail['width'];
            $h = (int)$detail['height'];
            if ($w <= 0 || $h <= 0) {
                throw new \Exception('Размеры деталей должны быть положительными');
            }
            // Проверка на габариты с учётом поворота
            if (($w > $this->sheetWidth && $w > $this->sheetHeight) ||
                ($h > $this->sheetWidth && $h > $this->sheetHeight)) {
                throw new \Exception("Деталь {$w}×{$h} слишком большая для листа {$this->sheetWidth}×{$this->sheetHeight}");
            }
            $this->details[] = [
                'id'     => $index++,
                'width'  => $w,
                'height' => $h,
                'area'   => $w * $h
            ];
        }

        // Сортируем детали по площади (от больших к маленьким) для ускорения раскладки
        usort($this->details, function($a, $b) {
            return $b['area'] - $a['area'];
        });
    }

    public function optimizeGA()
    {
        // Создаём начальную популяцию с разными стратегиями
        $population = $this->createInitialPopulation();

        $bestFitness    = PHP_FLOAT_MAX;
        $bestPlacements = [];
        $noImprovementCount = 0;
        $lastBestFitness = PHP_FLOAT_MAX;

        for ($gen = 0; $gen < $this->generations; $gen++) {
            $fitnesses = [];

            foreach ($population as $idx => $chromosome) {
                $chromosomeKey = implode(',', $chromosome);
                if (isset($this->fitnessCache[$chromosomeKey])) {
                    $fitness    = $this->fitnessCache[$chromosomeKey]['fitness'];
                    $placements = $this->fitnessCache[$chromosomeKey]['placements'];
                } else {
                    $placements = $this->decodeChromosome($chromosome);
                    $fitness    = $this->calculateFitness($placements);
                    $this->fitnessCache[$chromosomeKey] = [
                        'fitness'    => $fitness,
                        'placements' => $placements
                    ];
                    // Ограничиваем размер кэша (хранить не более 1000 записей)
                    if (count($this->fitnessCache) > 1000) {
                        array_shift($this->fitnessCache);
                    }
                }

                $fitnesses[$idx] = $fitness;
                if ($fitness < $bestFitness) {
                    $bestFitness    = $fitness;
                    $bestPlacements = $placements;
                }
            }

            // Проверка сходимости (если за 8 поколений нет значимого улучшения – выходим)
            if (abs($bestFitness - $lastBestFitness) < 0.001) {
                $noImprovementCount++;
                if ($noImprovementCount >= 8) {
                    break;
                }
            } else {
                $noImprovementCount = 0;
                $lastBestFitness    = $bestFitness;
            }

            // Формируем новое поколение
            $newPopulation = [];

            // 1) Элитизм – переносим лучших особей
            $eliteCount = max(1, (int)($this->populationSize * $this->elitismRate));
            $sortedIdx  = $this->sortPopulationByFitness($fitnesses);
            for ($i = 0; $i < $eliteCount; $i++) {
                $bestChromosomeIdx = $sortedIdx[$i];
                $newPopulation[]   = $population[$bestChromosomeIdx];
            }

            // 2) Остальное – через турнирную селекцию, кроссовер и мутацию
            while (count($newPopulation) < $this->populationSize) {
                $parent1 = $this->selectParent($population, $fitnesses);
                $parent2 = $this->selectParent($population, $fitnesses);

                list($child1, $child2) = $this->crossover($parent1, $parent2);
                $child1 = $this->mutate($child1);
                $child2 = $this->mutate($child2);

                $newPopulation[] = $child1;
                if (count($newPopulation) < $this->populationSize) {
                    $newPopulation[] = $child2;
                }
            }

            $population = $newPopulation;
        }

        // После последней итерации – пытаемся перераспределить детали, чтобы убрать лишние листы
        $bestPlacements = $this->redistributePlacements($bestPlacements);
        return $bestPlacements;
    }

    private function analyzeSpaceUtilization($placements)
    {
        // (тут просто собирается статистика по каждому листу, если захочешь логировать)
    }

    /**
     * Найти все свободные прямоугольники в дереве разбивки.
     * Уже отфильтровываем мелкие кусочки < minScrapWidth×minScrapHeight
     * (учитываем возможный поворот остатка).
     */
    private function findFreeSpaces($node)
    {
        $spaces = [];

        // Если узел полностью занят – возврат пустого массива
        if ($node['isOccupied']) {
            return $spaces;
        }

        // Если у узла нет детей – он целиком свободен
        if (empty($node['children'])) {
            // И сразу проверяем по минимальному размеру
            if (
                (
                    $node['width']  >= $this->minScrapWidth &&
                    $node['height'] >= $this->minScrapHeight
                ) || (
                    $node['width']  >= $this->minScrapHeight &&
                    $node['height'] >= $this->minScrapWidth
                )
            ) {
                return [[
                    'x'      => $node['x'],
                    'y'      => $node['y'],
                    'width'  => $node['width'],
                    'height' => $node['height']
                ]];
            } else {
                return [];
            }
        }

        // Иначе рекурсивно проходимся по всем детям и собираем их свободные области
        foreach ($node['children'] as $child) {
            $childSpaces = $this->findFreeSpaces($child);
            $spaces = array_merge($spaces, $childSpaces);
        }

        return $spaces;
    }

    private function calculateFreeSpacesSimple($sheetDetails, $sheetIndex)
    {
        // Если лист «большой» (> 3000×3000), используем дерево, но с фильтром в findFreeSpaces()
        if ($this->sheetWidth > 3000 || $this->sheetHeight > 3000) {
            $spaceMap = [
                'isOccupied' => false,
                'x' => 0,
                'y' => 0,
                'width' => $this->sheetWidth,
                'height' => $this->sheetHeight,
                'children' => []
            ];
            foreach ($sheetDetails as $detail) {
                $this->addDetailToSpaceMap($spaceMap, $detail);
            }
            return $this->findFreeSpaces($spaceMap);
        }

        // Для «маленьких» листов (< 500×500) оставляем bitmap-алгоритм (он тоже фильтрует в конце)
        $bitmap = array_fill(0, $this->sheetHeight, array_fill(0, $this->sheetWidth, false));
        foreach ($sheetDetails as $detail) {
            for ($y = $detail['y']; $y < $detail['y'] + $detail['height']; $y++) {
                for ($x = $detail['x']; $x < $detail['x'] + $detail['width']; $x++) {
                    if ($x >= 0 && $x < $this->sheetWidth && $y >= 0 && $y < $this->sheetHeight) {
                        $bitmap[$y][$x] = true;
                    }
                }
            }
        }
        return $this->getMaximalEmptyRectangles($bitmap, $this->sheetWidth, $this->sheetHeight, $this->minScrapWidth, $this->minScrapHeight);
    }

    private function findMaxRectangle(&$bitmap, &$visited, $startX, $startY)
    {
        if ($visited[$startY][$startX]) {
            return ['x' => $startX, 'y' => $startY, 'width' => 0, 'height' => 0];
        }

        $maxWidth  = 0;
        $maxHeight = 0;
        $bestArea  = 0;
        $bestRect  = ['x' => $startX, 'y' => $startY, 'width' => 1, 'height' => 1];

        // Определяем максимально возможную ширину
        for ($x = $startX; $x < $this->sheetWidth && !$bitmap[$startY][$x]; $x++) {
            $maxWidth++;
        }

        // Для каждой возможной ширины выясняем, насколько можно растянуть высоту
        for ($width = 1; $width <= $maxWidth; $width++) {
            $height = 0;
            for ($y = $startY; $y < $this->sheetHeight; $y++) {
                $canExtend = true;
                for ($x = $startX; $x < $startX + $width; $x++) {
                    if ($bitmap[$y][$x]) {
                        $canExtend = false;
                        break;
                    }
                }
                if (!$canExtend) {
                    break;
                }
                $height++;
            }
            $area = $width * $height;
            if ($area > $bestArea) {
                $bestArea = $area;
                $bestRect = [
                    'x'      => $startX,
                    'y'      => $startY,
                    'width'  => $width,
                    'height' => $height
                ];
            }
        }

        // Помечаем найденную область как «посещённую»
        for ($y = $bestRect['y']; $y < $bestRect['y'] + $bestRect['height']; $y++) {
            for ($x = $bestRect['x']; $x < $bestRect['x'] + $bestRect['width']; $x++) {
                $visited[$y][$x] = true;
            }
        }

        return $bestRect;
    }

    private function getMaximalEmptyRectangles($bitmap, $width, $height, $minW, $minH)
    {
        $rects    = [];
        $rowAbove = array_fill(0, $width, 0);

        for ($y = 0; $y < $height; $y++) {
            $row = array_fill(0, $width, 0);
            for ($x = 0; $x < $width; $x++) {
                if (!$bitmap[$y][$x]) {
                    $row[$x] = ($y == 0 ? 1 : $rowAbove[$x] + 1);
                }
            }
            for ($x1 = 0; $x1 < $width; $x1++) {
                if ($row[$x1] == 0) continue;
                $minHNow = $row[$x1];
                for ($x2 = $x1; $x2 < $width && $row[$x2] > 0; $x2++) {
                    $minHNow = min($minHNow, $row[$x2]);
                    $w = $x2 - $x1 + 1;
                    if (
                        ($w >= $minW && $minHNow >= $minH) ||
                        ($w >= $minH && $minHNow >= $minW)
                    ) {
                        $rects[] = [
                            'x'      => $x1,
                            'y'      => $y - $minHNow + 1,
                            'width'  => $w,
                            'height' => $minHNow
                        ];
                    }
                }
            }
            $rowAbove = $row;
        }

        return $rects;
    }

    /**
     * Перераспределить детали с более высоких листов на свободные пространства предыдущих.
     */
    private function redistributePlacements($placements)
    {
        if (count($placements) < 10) {
            return $placements;
        }

        $maxSheet = 0;
        foreach ($placements as $p) {
            $maxSheet = max($maxSheet, $p['sheet']);
        }
        if ($maxSheet == 0) {
            return $placements;
        }

        // Организуем детали по листам (индексы в массиве placements)
        $sheetDetails = [];
        for ($i = 0; $i <= $maxSheet; $i++) {
            $sheetDetails[$i] = [];
        }
        foreach ($placements as $idx => $p) {
            $sheetDetails[$p['sheet']][] = $idx;
        }

        $maxSheetsToAnalyze = min($maxSheet, 50);

        // Построим деревья свободных пространств для каждого листа
        $sheetSpaceMaps    = [];
        $freeSpacesBySheet = [];
        for ($sheet = 0; $sheet < $maxSheetsToAnalyze; $sheet++) {
            $spaceMap = [
                'isOccupied' => false,
                'x' => 0,
                'y' => 0,
                'width' => $this->sheetWidth,
                'height' => $this->sheetHeight,
                'children' => []
            ];
            foreach ($sheetDetails[$sheet] as $detailIdx) {
                $this->addDetailToSpaceMap($spaceMap, $placements[$detailIdx]);
            }
            $freeSpaces = $this->findFreeSpaces($spaceMap);
            usort($freeSpaces, function($a, $b) {
                return ($b['width'] * $b['height']) - ($a['width'] * $a['height']);
            });
            $freeSpaces = array_slice($freeSpaces, 0, 20);
            $sheetSpaceMaps[$sheet]    = $spaceMap;
            $freeSpacesBySheet[$sheet] = $freeSpaces;
        }

        $moved     = true;
        $iterations = 0;
        while ($moved && $iterations < 5) {
            $moved     = false;
            $iterations++;

            for ($sheet = $maxSheet; $sheet > 0; $sheet--) {
                if (empty($sheetDetails[$sheet])) {
                    continue;
                }
                foreach ($sheetDetails[$sheet] as $indexInList => $detailIdx) {
                    $detail = $placements[$detailIdx];
                    $w      = $detail['width'];
                    $h      = $detail['height'];
                    // Пробуем переместить эту деталь на любой предыдущий лист
                    for ($targetSheet = 0; $targetSheet < $sheet && $targetSheet < $maxSheetsToAnalyze; $targetSheet++) {
                        if (empty($freeSpacesBySheet[$targetSheet])) {
                            continue;
                        }
                        foreach ($freeSpacesBySheet[$targetSheet] as $spaceIdx => $space) {
                            if ($space['width'] < 10 || $space['height'] < 10) {
                                continue;
                            }
                            $canPlace     = false;
                            $needRotation = false;
                            if ($w <= $space['width'] && $h <= $space['height']) {
                                $canPlace = true;
                            } elseif ($h <= $space['width'] && $w <= $space['height']) {
                                $canPlace     = true;
                                $needRotation = true;
                            }
                            if ($canPlace) {
                                if ($needRotation) {
                                    $tmp = $w;
                                    $w   = $h;
                                    $h   = $tmp;
                                }
                                // Переместили
                                $placements[$detailIdx]['sheet'] = $targetSheet;
                                $placements[$detailIdx]['x']     = $space['x'];
                                $placements[$detailIdx]['y']     = $space['y'];
                                $placements[$detailIdx]['width']  = $w;
                                $placements[$detailIdx]['height'] = $h;
                                $placements[$detailIdx]['rotation'] =
                                    $needRotation ? !$detail['rotation'] : $detail['rotation'];

                                // Убираем из старого листа
                                unset($sheetDetails[$sheet][$indexInList]);
                                $sheetDetails[$targetSheet][] = $detailIdx;

                                // Пересоздаём дерево и список свободных областей целевого листа
                                if ($targetSheet < $maxSheetsToAnalyze) {
                                    $spaceMap = $sheetSpaceMaps[$targetSheet];
                                    $this->addDetailToSpaceMap($spaceMap, $placements[$detailIdx]);
                                    $freeSpaces = $this->findFreeSpaces($spaceMap);
                                    usort($freeSpaces, function($a, $b) {
                                        return ($b['width'] * $b['height']) - ($a['width'] * $a['height']);
                                    });
                                    $freeSpaces = array_slice($freeSpaces, 0, 20);
                                    $sheetSpaceMaps[$targetSheet]    = $spaceMap;
                                    $freeSpacesBySheet[$targetSheet] = $freeSpaces;
                                }

                                $moved = true;
                                break 2;
                            }
                        }
                    }
                }

                // Если на листе больше нет деталей, «сжимаем» номера листов
                if (empty($sheetDetails[$sheet])) {
                    for ($s = $sheet + 1; $s <= $maxSheet; $s++) {
                        foreach ($sheetDetails[$s] as $detailIdx) {
                            $placements[$detailIdx]['sheet']--;
                        }
                        $sheetDetails[$s - 1] = $sheetDetails[$s];
                    }
                    unset($sheetDetails[$maxSheet]);
                    $maxSheet--;
                }
            }
        }

        return $placements;
    }

    public function getStatistics($placements)
    {
        if (empty($placements)) {
            return [
                'sheetsCount' => 0,
                'totalArea'   => 0,
                'usedArea'    => 0,
                'efficiency'  => 0,
                'wasteArea'   => 0,
                'usableScrap' => 0
            ];
        }

        $maxSheet   = 0;
        $usedArea   = 0;
        $usableScrap = 0;
        $sheetDetails = [];

        foreach ($placements as $p) {
            $usedArea += $p['width'] * $p['height'];
            $maxSheet = max($maxSheet, $p['sheet']);
            if (!isset($sheetDetails[$p['sheet']])) {
                $sheetDetails[$p['sheet']] = [];
            }
            $sheetDetails[$p['sheet']][] = [
                'x'      => $p['x'],
                'y'      => $p['y'],
                'width'  => $p['width'],
                'height' => $p['height']
            ];
        }

        // Для каждого листа считаем свободные области и суммируем те, что ≥ minScrap
        for ($i = 0; $i <= $maxSheet; $i++) {
            if (!isset($sheetDetails[$i])) {
                $sheetDetails[$i] = [];
            }
            $freeSpaces = $this->calculateFreeSpacesSimple($sheetDetails[$i], $i);
            foreach ($freeSpaces as $space) {
                $w = $space['width'];
                $h = $space['height'];
                error_log("Остаток: {$w} × {$h}");
                if (
                    ($w >= $this->minScrapWidth && $h >= $this->minScrapHeight) ||
                    ($w >= $this->minScrapHeight && $h >= $this->minScrapWidth)
                ) {
                    $usableScrap += $w * $h;
                }
            }
        }

        $sheetsCount = $maxSheet + 1;
        $totalArea   = $sheetsCount * $this->sheetWidth * $this->sheetHeight;
        $wasteArea   = $totalArea - $usedArea;
        $efficiency  = $totalArea > 0 ? ($usedArea / $totalArea) * 100 : 0;

        if ($usableScrap > $wasteArea) {
            $usableScrap = $wasteArea;
        }

        return [
            'sheetsCount' => $sheetsCount,
            'totalArea'   => $totalArea,
            'usedArea'    => $usedArea,
            'wasteArea'   => $wasteArea,
            'efficiency'  => $efficiency,
            'usableScrap' => $usableScrap
        ];
    }

    private function createInitialPopulation()
    {
        $population = [];
        $base       = range(0, count($this->details) - 1);

        // Несколько жадных начальных хромосом, сортируя по разным критериям
        $widthSorted  = $base;
        $heightSorted = $base;
        $areaSorted   = $base;
        $perimetrSorted = $base;

        usort($widthSorted, function($a, $b) {
            return $this->details[$b]['width'] - $this->details[$a]['width'];
        });
        usort($heightSorted, function($a, $b) {
            return $this->details[$b]['height'] - $this->details[$a]['height'];
        });
        usort($areaSorted, function($a, $b) {
            return $this->details[$b]['area'] - $this->details[$a]['area'];
        });
        usort($perimetrSorted, function($a, $b) {
            $pa = 2 * ($this->details[$a]['width'] + $this->details[$a]['height']);
            $pb = 2 * ($this->details[$b]['width'] + $this->details[$b]['height']);
            return $pb - $pa;
        });

        $population[] = $widthSorted;
        $population[] = $heightSorted;
        $population[] = $areaSorted;
        $population[] = $perimetrSorted;

        $aspectRatioSorted = $base;
        usort($aspectRatioSorted, function($a, $b) {
            $ra = max($this->details[$a]['width'], $this->details[$a]['height']) /
                  min($this->details[$a]['width'], $this->details[$a]['height']);
            $rb = max($this->details[$b]['width'], $this->details[$b]['height']) /
                  min($this->details[$b]['width'], $this->details[$b]['height']);
            return $rb - $ra;
        });
        $population[] = $aspectRatioSorted;

        // Остальные – случайные перестановки
        for ($i = 5; $i < $this->populationSize; $i++) {
            $chromosome = $base;
            shuffle($chromosome);
            $population[] = $chromosome;
        }

        return $population;
    }

    private function decodeChromosome($chromosome)
    {
        // Распаковываем детали в том порядке, в котором указан хромосомой
        $ordered = [];
        foreach ($chromosome as $idx) {
            $ordered[] = $this->details[$idx];
        }

        $placements = [];
        $sheets     = [];
        $sheetIndex = 0;
        $unplaced   = $ordered;

        while (!empty($unplaced)) {
            // Начинаем новый лист со всего свободного пространства
            $sheets[$sheetIndex] = [
                'spaces' => [[
                    'x'      => 0,
                    'y'      => 0,
                    'width'  => $this->sheetWidth,
                    'height' => $this->sheetHeight
                ]]
            ];

            $placedInThisSheet = true;
            while ($placedInThisSheet) {
                $placedInThisSheet = false;

                // Сортируем текущие свободные пространства по убыванию площади
                usort($sheets[$sheetIndex]['spaces'], function($a, $b) {
                    return ($b['width'] * $b['height']) - ($a['width'] * $a['height']);
                });

                // Для каждого пространства ищем, какую деталь туда впихнуть
                foreach ($sheets[$sheetIndex]['spaces'] as $spaceIndex => $space) {
                    $bestDetail      = null;
                    $bestFitScore    = INF;
                    $bestDetailIndex = -1;

                    foreach ($unplaced as $idx => $detail) {
                        foreach ([false, true] as $rotation) {
                            $w = $rotation ? $detail['height'] : $detail['width'];
                            $h = $rotation ? $detail['width'] : $detail['height'];
                            if ($w <= $space['width'] && $h <= $space['height']) {
                                // Вычисляем «коэффициент заполнения» по нескольким критериям
                                $areaLoss = ($space['width'] - $w) * ($space['height'] - $h);
                                $fitScore = min(
                                    abs($space['width'] - $w),
                                    abs($space['height'] - $h)
                                );
                                $cornerPreference = ($space['x'] == 0 || $space['y'] == 0) ? 0.9 : 1.0;
                                $combinedScore = $areaLoss * $cornerPreference + $fitScore * 0.1;
                                if ($combinedScore < $bestFitScore) {
                                    $bestFitScore    = $combinedScore;
                                    $bestDetail      = [
                                        'id'             => $detail['id'],
                                        'x'              => $space['x'],
                                        'y'              => $space['y'],
                                        'width'          => $w,
                                        'height'         => $h,
                                        'rotation'       => $rotation,
                                        'sheet'          => $sheetIndex,
                                        'originalWidth'  => $detail['width'],
                                        'originalHeight' => $detail['height']
                                    ];
                                    $bestDetailIndex = $idx;
                                }
                            }
                        }
                    }

                    if ($bestDetail !== null) {
                        // Размещаем самую «лучшую» деталь
                        $placements[] = $bestDetail;
                        array_splice($unplaced, $bestDetailIndex, 1);

                        // Обновляем список свободных пространств (разрезаем текущее)
                        $this->updateSpaces(
                            $sheets[$sheetIndex]['spaces'],
                            $spaceIndex,
                            $bestDetail['width'],
                            $bestDetail['height']
                        );
                        $placedInThisSheet = true;
                        break;
                    }
                }
            }
            $sheetIndex++;
        }

        return $placements;
    }

    private function calculateFitness($placements)
    {
        if (empty($placements)) {
            return PHP_FLOAT_MAX;
        }

        $maxSheet = 0;
        $usedArea = 0;
        $sheetDetails = [];

        foreach ($placements as $p) {
            $usedArea += $p['width'] * $p['height'];
            $maxSheet = max($maxSheet, $p['sheet']);
            if (!isset($sheetDetails[$p['sheet']])) {
                $sheetDetails[$p['sheet']] = [];
            }
            $sheetDetails[$p['sheet']][] = $p;
        }

        $sheetsCount = $maxSheet + 1;
        $totalArea   = $sheetsCount * $this->sheetWidth * $this->sheetHeight;
        $wasteArea   = $totalArea - $usedArea;

        // Базовый фитнесс = площадь отходов + штраф за каждый лист
        $fitness     = $wasteArea;
        $sheetPenalty = $sheetsCount * 5000;

        // Дополнительный штраф, если последний лист заполнен менее чем на 30%
        if (isset($sheetDetails[$maxSheet])) {
            $lastSheetUsedArea = 0;
            foreach ($sheetDetails[$maxSheet] as $detail) {
                $lastSheetUsedArea += $detail['width'] * $detail['height'];
            }
            $lastSheetEfficiency = $lastSheetUsedArea / ($this->sheetWidth * $this->sheetHeight);
            if ($lastSheetEfficiency < 0.3) {
                $fitness += (0.3 - $lastSheetEfficiency) * 50000;
            }
        }

        return $fitness + $sheetPenalty;
    }

    private function sortPopulationByFitness($fitnesses)
    {
        $indices = array_keys($fitnesses);
        usort($indices, function($a, $b) use ($fitnesses) {
            return $fitnesses[$a] <=> $fitnesses[$b];
        });
        return $indices;
    }

    private function selectParent($population, $fitnesses)
    {
        $bestIndex   = null;
        $bestFitness = PHP_FLOAT_MAX;

        for ($i = 0; $i < $this->tournamentSize; $i++) {
            $r = array_rand($population);
            if ($fitnesses[$r] < $bestFitness) {
                $bestFitness = $fitnesses[$r];
                $bestIndex   = $r;
            }
        }

        return $population[$bestIndex];
    }

    private function crossover($parent1, $parent2)
    {
        $size  = count($parent1);
        $start = rand(0, $size - 1);
        $end   = rand($start, $size - 1);

        $child1 = array_fill(0, $size, -1);
        $child2 = array_fill(0, $size, -1);

        // Копируем отрезок из каждого родителя
        for ($i = $start; $i <= $end; $i++) {
            $child1[$i] = $parent1[$i];
            $child2[$i] = $parent2[$i];
        }

        // Заполняем оставшиеся позиции (метод OX – order crossover)
        $this->fillRemaining($child1, $parent2, $start, $end);
        $this->fillRemaining($child2, $parent1, $start, $end);

        return [$child1, $child2];
    }

    private function fillRemaining(&$child, $parent, $start, $end)
    {
        $size      = count($parent);
        $pos       = ($end + 1) % $size;
        $parentPos = ($end + 1) % $size;

        $used = array_fill(0, $size, false);
        for ($i = $start; $i <= $end; $i++) {
            $used[$child[$i]] = true;
        }

        while (in_array(-1, $child, true)) {
            if ($child[$pos] === -1) {
                while ($used[$parent[$parentPos]]) {
                    $parentPos = ($parentPos + 1) % $size;
                }
                $child[$pos] = $parent[$parentPos];
                $used[$parent[$parentPos]] = true;
                $parentPos = ($parentPos + 1) % $size;
            }
            $pos = ($pos + 1) % $size;
        }
    }

    private function mutate($chromosome)
    {
        // Случайный swap
        if (mt_rand() / mt_getrandmax() < $this->mutationRate) {
            $a     = rand(0, count($chromosome) - 1);
            $b     = rand(0, count($chromosome) - 1);
            $tmp   = $chromosome[$a];
            $chromosome[$a] = $chromosome[$b];
            $chromosome[$b] = $tmp;
        }
        // Иногда делаем небольшую инверсию подотрезка
        if (mt_rand() / mt_getrandmax() < 0.05) {
            $size  = count($chromosome);
            $start = rand(0, $size - 2);
            $end   = rand($start + 1, $size - 1);
            $sub   = array_slice($chromosome, $start, $end - $start + 1);
            $sub   = array_reverse($sub);
            array_splice($chromosome, $start, $end - $start + 1, $sub);
        }

        return $chromosome;
    }

    /**
     * Разрезаем текущее свободное пространство на два (горизонтально или вертикально),
     * в зависимости от того, как образуются более «полезные» остатки.
     */
    private function updateSpaces(&$spaces, $usedIndex, $w, $h)
    {
        $currentSpace = $spaces[$usedIndex];
        unset($spaces[$usedIndex]);
        $spaces = array_values($spaces);

        $remainingW = $currentSpace['width'] - $w;
        $remainingH = $currentSpace['height'] - $h;

        // Проверяем, с какими остатками будет лучше (горизонтально или вертикально)
        $isRemainingWUseful = ($remainingW >= $this->minScrapWidth && $h >= $this->minScrapHeight) ||
                              ($remainingW >= $this->minScrapHeight && $h >= $this->minScrapWidth);
        $isRemainingHUseful = ($currentSpace['width'] >= $this->minScrapWidth && $remainingH >= $this->minScrapHeight) ||
                              ($currentSpace['width'] >= $this->minScrapHeight && $remainingH >= $this->minScrapWidth);

        $horizontalUsefulSpaces = 0;
        $verticalUsefulSpaces   = 0;

        if ($remainingW > 0 && $isRemainingWUseful) {
            $horizontalUsefulSpaces++;
        }
        if ($remainingH > 0 && $isRemainingHUseful) {
            $horizontalUsefulSpaces++;
        }

        $isVerticalWUseful = ($remainingW >= $this->minScrapWidth && $currentSpace['height'] >= $this->minScrapHeight) ||
                             ($remainingW >= $this->minScrapHeight && $currentSpace['height'] >= $this->minScrapWidth);
        $isVerticalHUseful = ($w >= $this->minScrapWidth && $remainingH >= $this->minScrapHeight) ||
                             ($w >= $this->minScrapHeight && $remainingH >= $this->minScrapWidth);

        if ($remainingH > 0 && $isVerticalHUseful) {
            $verticalUsefulSpaces++;
        }
        if ($remainingW > 0 && $isVerticalWUseful) {
            $verticalUsefulSpaces++;
        }

        if ($horizontalUsefulSpaces == $verticalUsefulSpaces) {
            // Рассчитываем, какая остаточная площадь будет меньше «бесполезной»
            $wasteH = ($remainingW > 0 && !$isRemainingWUseful) ? $remainingW * $h : 0;
            $wasteH += ($remainingH > 0 && !$isRemainingHUseful) ? $remainingH * $currentSpace['width'] : 0;

            $wasteV = ($remainingH > 0 && !$isVerticalHUseful) ? $remainingH * $w : 0;
            $wasteV += ($remainingW > 0 && !$isVerticalWUseful) ? $remainingW * $currentSpace['height'] : 0;

            $useHorizontal = $wasteH <= $wasteV;
        } else {
            $useHorizontal = $horizontalUsefulSpaces > $verticalUsefulSpaces;
        }

        if ($useHorizontal) {
            // Оставляем справа остаток шириной remainingW и высотой h
            if ($remainingW > 0) {
                $spaces[] = [
                    'x'      => $currentSpace['x'] + $w,
                    'y'      => $currentSpace['y'],
                    'width'  => $remainingW,
                    'height' => $h
                ];
            }
            // Оставляем снизу остаток на всю ширину currentSpace['width']
            if ($remainingH > 0) {
                $spaces[] = [
                    'x'      => $currentSpace['x'],
                    'y'      => $currentSpace['y'] + $h,
                    'width'  => $currentSpace['width'],
                    'height' => $remainingH
                ];
            }
        } else {
            // Оставляем снизу кусочек шириной w
            if ($remainingH > 0) {
                $spaces[] = [
                    'x'      => $currentSpace['x'],
                    'y'      => $currentSpace['y'] + $h,
                    'width'  => $w,
                    'height' => $remainingH
                ];
            }
            // Оставляем справа остаток на всю высоту currentSpace['height']
            if ($remainingW > 0) {
                $spaces[] = [
                    'x'      => $currentSpace['x'] + $w,
                    'y'      => $currentSpace['y'],
                    'width'  => $remainingW,
                    'height' => $currentSpace['height']
                ];
            }
        }

        // Сливаем смежные или вложенные остатки, чтобы не фрагментировать пространство
        $this->mergeOverlappingSpaces($spaces);

        // Сортируем по координате (сначала верхние, потом левые), чтобы стабильнее проходить рекурсию
        usort($spaces, function($a, $b) {
            return ($a['y'] * 10000 + $a['x']) - ($b['y'] * 10000 + $b['x']);
        });
    }

    private function mergeOverlappingSpaces(&$spaces)
    {
        $merged = true;
        while ($merged && count($spaces) > 1) {
            $merged = false;
            $n = count($spaces);
            for ($i = 0; $i < $n; $i++) {
                for ($j = $i + 1; $j < $n; $j++) {
                    // Если один прямоугольник вложен в другой — убираем вложенный
                    if ($this->isContained($spaces[$i], $spaces[$j])) {
                        unset($spaces[$j]);
                        $spaces = array_values($spaces);
                        $merged = true;
                        break 2;
                    } elseif ($this->isContained($spaces[$j], $spaces[$i])) {
                        unset($spaces[$i]);
                        $spaces = array_values($spaces);
                        $merged = true;
                        break 2;
                    }

                    // Если можно объединить два смежных (горизонтально или вертикально)
                    $mergedSpace = $this->tryMergeSpaces($spaces[$i], $spaces[$j]);
                    if ($mergedSpace !== false) {
                        $spaces[$i] = $mergedSpace;
                        unset($spaces[$j]);
                        $spaces = array_values($spaces);
                        $merged = true;
                        break 2;
                    }
                }
            }
        }
    }

    private function tryMergeSpaces($a, $b)
    {
        // Горизонтальное объединение: одинаковая высота, соседние по x
        if ($a['y'] == $b['y'] && $a['height'] == $b['height']) {
            if ($a['x'] + $a['width'] == $b['x']) {
                return [
                    'x'      => $a['x'],
                    'y'      => $a['y'],
                    'width'  => $a['width'] + $b['width'],
                    'height' => $a['height']
                ];
            }
            if ($b['x'] + $b['width'] == $a['x']) {
                return [
                    'x'      => $b['x'],
                    'y'      => $b['y'],
                    'width'  => $b['width'] + $a['width'],
                    'height' => $b['height']
                ];
            }
        }

        // Вертикальное объединение: одинаковая ширина, соседние по y
        if ($a['x'] == $b['x'] && $a['width'] == $b['width']) {
            if ($a['y'] + $a['height'] == $b['y']) {
                return [
                    'x'      => $a['x'],
                    'y'      => $a['y'],
                    'width'  => $a['width'],
                    'height' => $a['height'] + $b['height']
                ];
            }
            if ($b['y'] + $b['height'] == $a['y']) {
                return [
                    'x'      => $b['x'],
                    'y'      => $b['y'],
                    'width'  => $b['width'],
                    'height' => $b['height'] + $a['height']
                ];
            }
        }

        return false;
    }

    private function isContained($a, $b)
    {
        return $b['x'] >= $a['x'] &&
               $b['y'] >= $a['y'] &&
               $b['x'] + $b['width']  <= $a['x'] + $a['width'] &&
               $b['y'] + $b['height'] <= $a['y'] + $a['height'];
    }

    /**
     * Рекурсивно добавляем одну деталь (detail) в дерево свободных областей (node),
     * «закрашивая» её и разбивая родительский узел на 4 дочерних, если надо.
     */
    private function addDetailToSpaceMap(&$node, $detail)
    {
        if ($node['isOccupied']) {
            return;
        }
        if (!$this->intersects(
            $node['x'], $node['y'], $node['width'], $node['height'],
            $detail['x'], $detail['y'], $detail['width'], $detail['height']
        )) {
            return;
        }
        // Если detail полностью закрывает текущий узел
        if (
            $detail['x'] <= $node['x'] &&
            $detail['y'] <= $node['y'] &&
            $detail['x'] + $detail['width']  >= $node['x'] + $node['width'] &&
            $detail['y'] + $detail['height'] >= $node['y'] + $node['height']
        ) {
            $node['isOccupied'] = true;
            $node['children']   = [];
            return;
        }

        // Если узел слишком мал (меньше, чем 10×10), просто помечаем занят
        if ($node['width'] < 10 || $node['height'] < 10) {
            $node['isOccupied'] = true;
            $node['children']   = [];
            return;
        }

        // Разбиваем на 4 подузла, если ещё не разбивали
        if (empty($node['children'])) {
            $splitX = floor($node['width'] / 2);
            $splitY = floor($node['height'] / 2);

            // Делаем «точный» разрез по границам detail, если он пересекает
            if ($detail['x'] > $node['x'] && $detail['x'] < $node['x'] + $node['width']) {
                $splitX = $detail['x'] - $node['x'];
            } elseif (
                $detail['x'] + $detail['width'] > $node['x'] &&
                $detail['x'] + $detail['width'] < $node['x'] + $node['width']
            ) {
                $splitX = $detail['x'] + $detail['width'] - $node['x'];
            }

            if ($detail['y'] > $node['y'] && $detail['y'] < $node['y'] + $node['height']) {
                $splitY = $detail['y'] - $node['y'];
            } elseif (
                $detail['y'] + $detail['height'] > $node['y'] &&
                $detail['y'] + $detail['height'] < $node['y'] + $node['height']
            ) {
                $splitY = $detail['y'] + $detail['height'] - $node['y'];
            }

            $splitX = max(1, min($splitX, $node['width'] - 1));
            $splitY = max(1, min($splitY, $node['height'] - 1));

            $node['children'] = [
                // Верхний левый
                [
                    'isOccupied' => false,
                    'x' => $node['x'],
                    'y' => $node['y'],
                    'width' => $splitX,
                    'height' => $splitY,
                    'children' => []
                ],
                // Верхний правый
                [
                    'isOccupied' => false,
                    'x' => $node['x'] + $splitX,
                    'y' => $node['y'],
                    'width' => $node['width'] - $splitX,
                    'height' => $splitY,
                    'children' => []
                ],
                // Нижний левый
                [
                    'isOccupied' => false,
                    'x' => $node['x'],
                    'y' => $node['y'] + $splitY,
                    'width' => $splitX,
                    'height' => $node['height'] - $splitY,
                    'children' => []
                ],
                // Нижний правый
                [
                    'isOccupied' => false,
                    'x' => $node['x'] + $splitX,
                    'y' => $node['y'] + $splitY,
                    'width' => $node['width'] - $splitX,
                    'height' => $node['height'] - $splitY,
                    'children' => []
                ]
            ];
        }

        // Рекурсивно «закрашиваем» детей
        foreach ($node['children'] as &$child) {
            $this->addDetailToSpaceMap($child, $detail);
        }

        // Если все 4 ребёнка теперь заняты — помечаем их родителя как «occupied»
        $allOccupied = true;
        foreach ($node['children'] as $child) {
            if (!$child['isOccupied']) {
                $allOccupied = false;
                break;
            }
        }
        if ($allOccupied) {
            $node['isOccupied'] = true;
            $node['children']   = [];
        }
    }

    private function intersects($x1, $y1, $w1, $h1, $x2, $y2, $w2, $h2)
    {
        return !(
            $x1 + $w1 <= $x2 ||
            $x2 + $w2 <= $x1 ||
            $y1 + $h1 <= $y2 ||
            $y2 + $h2 <= $y1
        );
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    try {
        $rawData = file_get_contents('php://input');
        $data    = json_decode($rawData, true);

        if (!$data) {
            throw new \Exception('Не удалось декодировать JSON данные');
        }
        if (!isset($data['sheetWidth'], $data['sheetHeight'], $data['details'])) {
            throw new \Exception('Отсутствуют обязательные параметры: sheetWidth, sheetHeight, details');
        }

        $sheetWidth  = (int)$data['sheetWidth'];
        $sheetHeight = (int)$data['sheetHeight'];
        $details     = $data['details'];

        if (empty($details)) {
            throw new \Exception('Список деталей пуст');
        }
        if ($sheetWidth <= 0 || $sheetHeight <= 0) {
            throw new \Exception('Размеры листа должны быть положительными числами');
        }

        $startTime = microtime(true);

        $options = [];
        if (isset($data['algorithmSettings'])) {
            $options['algorithmSettings'] = $data['algorithmSettings'];
        }

        $optimizer  = new GeneticCuttingOptimizer($sheetWidth, $sheetHeight, $details, $options);
        $placements = $optimizer->optimizeGA();
        $stats      = $optimizer->getStatistics($placements);

        $executionTime       = microtime(true) - $startTime;
        $stats['executionTime'] = round($executionTime, 2);

        echo json_encode([
            'success' => true,
            'data'    => $placements,
            'stats'   => $stats
        ]);

    } catch (Exception $e) {
        error_log('Ошибка в оптимизаторе раскроя: ' . $e->getMessage());
        http_response_code(500);
        echo json_encode([
            'success' => false,
            'message' => $e->getMessage()
        ]);
    }
}
?>
